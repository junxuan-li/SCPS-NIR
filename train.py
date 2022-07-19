import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import glob
import matplotlib.pyplot as plt
import cv2 as cv
import yaml
import math
import argparse

from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile

from models import totalVariation, Spherical_Gaussian, totalVariation_L2, Light_Model, Light_Model_CNN, NeRFModel_Separate

from cfgnode import CfgNode
from dataloader import Data_Loader
from load_diligent import load_diligent
from load_lightstage import load_lightstage
from load_apple import load_apple
from load_sythn import load_sythn
from position_encoder import get_embedding_function

from utils import writer_add_image

from utils import cal_ints_acc, cal_dirs_acc, add_noise_light_init
from draw_utils import plot_lighting, plot_dir_error, plot_int_error, plot_lighting_gt
from dynamic_basis import dynamic_basis


def train(input_data, testing):
    batch_size = input_data['rgb'].size(0)
    input_xy = input_data['input_xy'][0].to(device)
    gt_normal = input_data['normal'][0].to(device)

    gt_rgb = input_data['rgb'].view(-1, 1 if cfg.dataset.gray_scale else 3).to(device)
    gt_shadow_mask = input_data['shadow_mask'].view(-1, 1).to(device)
    if cfg.loss.contour_factor > 0:
        pre_contour_normal = input_data['contour_normal'][0].to(device)

    embed_input = encode_fn_input1(input_xy)
    if cfg.models.use_mean_var:
        mean_var = input_data['mean_var'][0].to(device)
        embed_input = torch.cat([embed_input, mean_var], dim=-1)

    output_normal_0, output_diff_0, output_spec_coeff_0 = model(embed_input)
    output_normal = output_normal_0.repeat(batch_size, 1)
    output_diff = output_diff_0.repeat(batch_size, 1)
    output_spec_coeff = output_spec_coeff_0.repeat(batch_size, 1)

    est_light_direction, est_light_intensity = light_model.get_light_from_idx(idx=input_data['item_idx'].to(device))
    input_light_direction = est_light_direction

    if cfg.models.specular.type == 'Spherical_Gaussian':
        output_spec_mu = output_spec_coeff[..., cfg.models.specular.num_basis:].view(-1,
                                                                                     cfg.models.specular.num_basis,
                                                                                     1 if cfg.dataset.gray_scale else 3)

        if hasattr(cfg.models.specular, 'dynamic_basis'):
            if cfg.models.specular.dynamic_basis and not configargs.testing:
                output_spec_mu = dynamic_basis(output_spec_mu, epoch, end_epoch, cfg.models.specular.num_basis)

        output_spe = specular_model(light=input_light_direction, normal=output_normal, mu=output_spec_mu)
        output_spe = output_spe.sum(dim=1)

    output_rho = output_diff + output_spe

    if hasattr(cfg.models, 'use_onlyspecular'):
        if cfg.models.use_onlyspecular:
            output_rho = output_spe

    if hasattr(cfg.models, 'use_onlydiffuse'):
        if cfg.models.use_onlydiffuse:
            output_rho = output_diff

    render_shading = F.relu((output_normal * input_light_direction).sum(dim=-1, keepdims=True))
    render_rgb = output_rho * render_shading
    render_rgb = render_rgb * est_light_intensity

    if not testing:
        rgb_loss = rgb_loss_function(render_rgb * gt_shadow_mask, gt_rgb * gt_shadow_mask)
        rgb_loss_val = rgb_loss.item()
        loss = rgb_loss

        if epoch <= int(cfg.loss.regularize_epoches * end_epoch):  # if epoch is small, use tv to guide the network
            if cfg.loss.diff_tv_factor > 0:
                diff_color_map = torch.zeros((h, w, 1 if cfg.dataset.gray_scale else 3), dtype=torch.float32, device=device)
                diff_color_map[idxp] = output_diff_0
                tv_loss = totalVariation(diff_color_map, mask, num_rays) * batch_size * cfg.loss.diff_tv_factor
                loss += tv_loss
            if cfg.loss.spec_tv_factor > 0:
                spec_color_map = torch.zeros((h, w, output_spec_coeff_0.size(1)), dtype=torch.float32, device=device)
                spec_color_map[idxp] = output_spec_coeff_0
                tv_loss = totalVariation(spec_color_map, mask, num_rays) * batch_size * cfg.loss.spec_tv_factor
                loss += tv_loss
            if cfg.loss.normal_tv_factor > 0:
                normal_map = torch.zeros((h, w, 3), dtype=torch.float32, device=device)
                normal_map[idxp] = output_normal_0
                tv_loss = totalVariation_L2(normal_map, mask, num_rays) * batch_size * cfg.loss.normal_tv_factor
                loss += tv_loss
            if cfg.loss.spec_coeff_factor > 0:
                spec_coeff_loss = F.l1_loss(output_spec_coeff_0, torch.zeros_like(output_spec_coeff_0))
                loss += spec_coeff_loss * cfg.loss.spec_coeff_factor * batch_size
        if cfg.loss.contour_factor > 0 and epoch <= int(0.75*end_epoch):
            contour_normal_loss = 1 - torch.sum(output_normal_0 * pre_contour_normal, dim=-1).mean()
            loss += contour_normal_loss * cfg.loss.contour_factor
            normal_map = torch.zeros((h, w, 3), dtype=torch.float32, device=device)
            normal_map[idxp] = output_normal_0
            tv_loss = totalVariation_L2(normal_map, mask, num_rays) * batch_size * cfg.loss.normal_tv_factor * cfg.loss.contour_factor
            loss += tv_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log the running loss
        normal_loss = torch.arccos(torch.clamp((output_normal_0 * gt_normal).sum(dim=-1), max=1, min=-1)).mean()
        cost_t = time.time() - start_t
        est_time = cost_t / ((epoch - start_epoch) * iters_per_epoch + iter_num + 1) * (
                (end_epoch - epoch) * iters_per_epoch + iters_per_epoch - iter_num - 1)
        print(
            'epoch: %d,  iter: %2d/ %d, Training: %.4f, Val: %.4f  cost_time: %d m %2d s,  est_time: %d m %2d s' %
            (epoch, iter_num + 1, iters_per_epoch, loss.item(), normal_loss.item() / math.pi * 180, cost_t // 60, cost_t % 60,
             est_time // 60, est_time % 60))
        writer.add_scalar('Training loss', rgb_loss_val, (epoch - 1) * iters_per_epoch + iter_num)
        writer.add_scalar('Val loss', normal_loss.item() / math.pi * 180, (epoch - 1) * iters_per_epoch + iter_num)

        pred_ld, pred_li = light_model.get_all_lights()
        ld_acc, _ = cal_dirs_acc(all_light_direction, pred_ld)
        li_acc, _ = cal_ints_acc(all_light_intensity, pred_li)
        # print("Light direction acc:  %.4f    Light intensity acc:  %.5f" % (ld_acc, li_acc))
        writer.add_scalar('Light direction acc', ld_acc, (epoch - 1) * iters_per_epoch + iter_num)
        writer.add_scalar('Light intensity acc', li_acc, (epoch - 1) * iters_per_epoch + iter_num)
    else:
        rgb_loss = F.l1_loss(render_rgb.view(-1), gt_rgb.view(-1))
        normal_loss = torch.arccos(torch.clamp((output_normal_0 * gt_normal).sum(dim=-1), max=1, min=-1)).mean()
        print("Testing RGB L1: %.4f" % (rgb_loss.item() * 255.))
        print("Normal MAE: %.4f" % (normal_loss.item() / math.pi * 180))

        if eval_idx == 1:
            normal_map = torch.zeros((h, w, 3), dtype=torch.float32, device=device)
            temp_nor = output_normal_0.clone()
            temp_nor[..., 1:] = -temp_nor[..., 1:]
            normal_map[idxp] = (temp_nor + 1) / 2
            normal_map = normal_map.cpu().numpy()
            normal_map = (np.clip(normal_map * 255., 0, 255)).astype(np.uint8)[:, :, ::-1]

            normal_map[idxp_invalid] = 255
            normal_map = normal_map[bounding_box_int[0]:bounding_box_int[1], bounding_box_int[2]:bounding_box_int[3]]
            cv.imwrite(os.path.join(log_path, 'est_normal.png'), normal_map)
            # create grid of images
            writer_add_image(os.path.join(log_path, 'est_normal.png'), epoch, writer)

            normal_map = torch.zeros((h, w, 3), dtype=torch.float32, device=device)
            normal_map[idxp] = temp_nor
            normal_map = normal_map.cpu().numpy()
            np.save(os.path.join(log_path, 'est_normal.npy'), normal_map)

            normalerr_map = torch.zeros((h, w), dtype=torch.float32, device=device)
            normal_err = torch.arccos(
                torch.clamp((output_normal_0 * gt_normal).sum(dim=-1), max=1, min=-1)) / math.pi * 180
            normalerr_map[idxp] = torch.clamp(normal_err, max=50)
            normalerr_map = normalerr_map.cpu().numpy()
            plt.matshow(normalerr_map)
            plt.colorbar()
            plt.savefig(os.path.join(log_path, 'est_normal_err.png'), dpi=200)
            plt.close()

            error_map = normalerr_map
            error_map = (np.clip(error_map / 50, 0, 1) * 255).astype(np.uint8)
            error_map = cv.applyColorMap(error_map, colormap=cv.COLORMAP_JET)

            error_map[idxp_invalid] = 255
            error_map = error_map[bounding_box_int[0]:bounding_box_int[1], bounding_box_int[2]:bounding_box_int[3]]
            cv.imwrite(os.path.join(log_path, 'est_normal_err_JET.png'), error_map)

            # create grid of images
            writer_add_image(os.path.join(log_path, 'est_normal_err.png'), epoch, writer)


            rgb_map = torch.ones((h, w, 1 if cfg.dataset.gray_scale else 3), dtype=torch.float32, device=device)
            rgb_map[idxp] = output_diff_0 / output_diff_0.max()
            rgb_map = rgb_map.cpu().numpy()
            rgb_map = np.clip(rgb_map * 255., 0., 255.).astype(np.uint8)[:, :, ::-1]
            rgb_map[idxp_invalid] = 255
            rgb_map = rgb_map[bounding_box_int[0]:bounding_box_int[1], bounding_box_int[2]:bounding_box_int[3]]
            cv.imwrite(os.path.join(log_path, 'est_brdf_diff.png'), rgb_map)
            # create grid of images
            writer_add_image(os.path.join(log_path, 'est_brdf_diff.png'), epoch, writer)

            for i in range(cfg.models.specular.num_basis):
                rgb_map = torch.zeros((h, w), dtype=torch.float32, device=device)
                rgb_map[idxp] = output_spec_coeff_0[:, i]
                rgb_map = rgb_map.cpu().numpy()
                plt.matshow(rgb_map)
                plt.colorbar()
                plt.savefig(os.path.join(log_path, 'est_brdf_speccoeff%d.png' % i), dpi=200)
                plt.close()

            if output_spec_coeff.size(1) > cfg.models.specular.num_basis:
                s_idx = cfg.models.specular.num_basis
                i = 0
                while s_idx < output_spec_coeff.size(1):
                    rgb_map = torch.zeros((h, w, 1 if cfg.dataset.gray_scale else 3), dtype=torch.float32, device=device)
                    rgb_map[idxp] = torch.abs(output_spec_coeff_0[:, s_idx:s_idx+(1 if cfg.dataset.gray_scale else 3)])
                    rgb_map = rgb_map.cpu().numpy()
                    rgb_map = np.clip(rgb_map * 255., 0., 255.).astype(np.uint8)
                    rgb_map[idxp_invalid] = 255
                    rgb_map = rgb_map[bounding_box_int[0]:bounding_box_int[1], bounding_box_int[2]:bounding_box_int[3]]
                    cv.imwrite(os.path.join(log_path, 'est_brdf_speccoeff%d_RGB.png' % i), rgb_map[:, :, ::-1])
                    s_idx = s_idx + (1 if cfg.dataset.gray_scale else 3)
                    i += 1

        if eval_idx >= 2:
            log_path_brdf_spec = os.path.join(log_path, 'brdf_spec')
            os.makedirs(log_path_brdf_spec, exist_ok=True)
            log_path_rgb = os.path.join(log_path, 'rgb')
            os.makedirs(log_path_rgb, exist_ok=True)
            log_path_brdf = os.path.join(log_path, 'brdf')
            os.makedirs(log_path_brdf, exist_ok=True)
            log_path_shading = os.path.join(log_path, 'shading')
            os.makedirs(log_path_shading, exist_ok=True)

        else:
            log_path_brdf_spec = log_path
            log_path_rgb = log_path
            log_path_brdf = log_path
            log_path_shading = log_path

        rgb_map = torch.ones((h, w, 1 if cfg.dataset.gray_scale else 3), dtype=torch.float32, device=device)
        rgb_map[idxp] = output_spe[:len(idxp[0])] * render_shading[:len(idxp[0])]
        rgb_map = rgb_map.cpu().numpy()
        rgb_map = np.clip(rgb_map * 255., 0., 255.).astype(np.uint8)[:, :, ::-1]
        rgb_map[idxp_invalid] = 255
        rgb_map = rgb_map[bounding_box_int[0]:bounding_box_int[1], bounding_box_int[2]:bounding_box_int[3]]
        cv.imwrite(os.path.join(log_path_brdf_spec, 'est_brdf_spec_%03d.png' % eval_idx), rgb_map)
        # create grid of images
        writer_add_image(os.path.join(log_path_brdf_spec, 'est_brdf_spec_%03d.png' % eval_idx), epoch, writer)

        ################ Get Specular on Unit Sphere ################
        if cfg.models.specular.type == 'Spherical_Gaussian':
            random_i = int(output_spec_mu.shape[0]*0.53)
            mu_random_point = output_spec_mu[random_i:random_i+1, :, :].repeat(unit_sphere_normal.shape[0], 1, 1)
            in_ld = torch.tensor([[0.3,-0.3,-0.90554]]).to(device)
            output_unit_spe_basis = specular_model(
                light=in_ld.repeat(unit_sphere_normal.size(0), 1),
                normal=unit_sphere_normal.to(device),
                # mu=torch.ones_like(unit_sphere_normal[:, None, :(1 if cfg.dataset.gray_scale else 3)].to(device)),
                mu=mu_random_point,
            )
        if cfg.models.specular.type == 'Spherical_Gaussian_Var' or cfg.models.specular.type == 'Microfacet_BRDF':
            output_unit_spe_basis = specular_model(
                light=input_light_direction[:1, :].repeat(unit_sphere_normal.size(0), 1),
                normal=unit_sphere_normal.to(device),
                k=torch.ones_like(unit_sphere_normal[..., :1].to(device))*(0.1),
                mu=torch.ones_like(unit_sphere_normal[:, None, :(1 if cfg.dataset.gray_scale else 3)].to(device)),
            )
            output_unit_spe_basis = output_unit_spe_basis[:, None, :]
        for basis_idx in range(output_unit_spe_basis.size(1)):
            rgb_map = torch.zeros((512, 612, 1 if cfg.dataset.gray_scale else 3), dtype=torch.float32, device=device)
            rgb_map[unit_sphere_idxp] = output_unit_spe_basis[:, basis_idx, :]
            rgb_map = rgb_map / rgb_map.max()
            rgb_map = rgb_map.cpu().numpy()
            rgb_map = np.clip(rgb_map * 255., 0., 255.).astype(np.uint8)[:, :, ::-1]
            rgb_map[unit_sphere_invalididxp] = 255
            rgb_map = rgb_map[unit_sphere_bounding_box_int[0]:unit_sphere_bounding_box_int[1], unit_sphere_bounding_box_int[2]:unit_sphere_bounding_box_int[3]]
            cv.imwrite(os.path.join(log_path_brdf_spec, 'est_brdf_spec_basis%03d_%03d.png' % (basis_idx, eval_idx)), rgb_map)
        rgb_map = torch.zeros((512, 612, 1 if cfg.dataset.gray_scale else 3), dtype=torch.float32, device=device)
        rgb_map[unit_sphere_idxp] = output_unit_spe_basis.sum(dim=1)
        rgb_map = rgb_map / rgb_map.max()
        rgb_map = rgb_map.cpu().numpy()
        rgb_map = np.clip(rgb_map * 255., 0., 255.).astype(np.uint8)[:, :, ::-1]
        rgb_map[unit_sphere_invalididxp] = 70
        rgb_map = rgb_map[unit_sphere_bounding_box_int[0]-5:unit_sphere_bounding_box_int[1]+5, unit_sphere_bounding_box_int[2]-5:unit_sphere_bounding_box_int[3]+5]
        cv.imwrite(os.path.join(log_path_brdf_spec, 'est_brdf_spec_basisSum_%03d.png' % eval_idx), rgb_map)
        ########################################################

        rgb_map = torch.zeros((h, w, 1 if cfg.dataset.gray_scale else 3), dtype=torch.float32, device=device)

        rgb_map[idxp] = render_rgb[:len(idxp[0])]
        rgb_map = rgb_map.cpu().numpy()
        rgb_map = np.clip(rgb_map * 255., 0., 255.).astype(np.uint8)[:, :, ::-1]
        rgb_map[idxp_invalid] = 255
        rgb_map = rgb_map[bounding_box_int[0]:bounding_box_int[1], bounding_box_int[2]:bounding_box_int[3]]
        cv.imwrite(os.path.join(log_path_rgb, 'est_rgb_%03d.png' % eval_idx), rgb_map)
        # create grid of images
        writer_add_image(os.path.join(log_path_rgb, 'est_rgb_%03d.png' % eval_idx), epoch, writer)

        rgb_map = torch.zeros((h, w, 1 if cfg.dataset.gray_scale else 3), dtype=torch.float32, device=device)
        rgb_map[idxp] = output_rho[:len(idxp[0])]
        rgb_map = rgb_map.cpu().numpy()
        rgb_map = np.clip(rgb_map * 255., 0., 255.).astype(np.uint8)[:, :, ::-1]
        rgb_map[idxp_invalid] = 255
        rgb_map = rgb_map[bounding_box_int[0]:bounding_box_int[1], bounding_box_int[2]:bounding_box_int[3]]
        cv.imwrite(os.path.join(log_path_brdf, 'est_brdf_%03d.png' % eval_idx), rgb_map)
        # create grid of images
        writer_add_image(os.path.join(log_path_rgb, 'est_brdf_%03d.png' % eval_idx), epoch, writer)


        rgb_map = torch.zeros((h, w, 1), dtype=torch.float32, device=device)
        rgb_map[idxp] = render_shading[:len(idxp[0])]
        rgb_map = rgb_map.cpu().numpy()
        rgb_map = np.clip(rgb_map * 255., 0., 255.).astype(np.uint8)[:, :, ::-1]
        rgb_map[idxp_invalid] = 255
        rgb_map = rgb_map[bounding_box_int[0]:bounding_box_int[1], bounding_box_int[2]:bounding_box_int[3]]
        cv.imwrite(os.path.join(log_path_shading, 'est_shading_%03d.png' % eval_idx), rgb_map)

        pred_ld, pred_li = light_model.get_all_lights()

        ld_acc, ld_err = cal_dirs_acc(all_light_direction.cpu(), pred_ld.cpu())
        li_acc, li_err = cal_ints_acc(all_light_intensity.cpu(), pred_li.cpu())
        print("Light direction acc:  %.4f" % ld_acc)
        print("Light intensity acc:  %.5f" % li_acc)
        plot_lighting(pred_ld.cpu().numpy(), pred_li.cpu().numpy(), log_path)
        plot_lighting_gt(all_light_direction.cpu().numpy(), all_light_intensity.cpu().numpy(), log_path)
        plot_dir_error(all_light_direction.cpu().numpy(), ld_err.cpu().numpy(), log_path)
        plot_int_error(all_light_direction.cpu().numpy(), li_err.cpu().numpy(), log_path)

        # create grid of images
        writer_add_image(os.path.join(log_path_rgb, 'est_light_map.png'), epoch, writer)

        np.savetxt(os.path.join(log_path, 'est_light_direction.txt'), pred_ld.cpu().numpy())
        np.savetxt(os.path.join(log_path, 'est_light_intensity.txt'), pred_li.cpu().numpy())

    return output_normal_0


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str,
        default="configs/diligent/reading.yml",
        help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--testing", type=str2bool,
        default=False,
        help="Enable testing mode."
    )
    parser.add_argument(
        "--cuda", type=str,
        help="Cuda ID."
    )
    parser.add_argument(
        "--quick_testing", type=str2bool,
        default=False,
        help="Enable quick_testing mode."
    )
    configargs = parser.parse_args()

    if configargs.quick_testing:
        configargs.testing = True

    # Read config file.
    configargs.config = os.path.expanduser(configargs.config)
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    if cfg.experiment.randomseed is not None:
        np.random.seed(cfg.experiment.randomseed)
        torch.manual_seed(cfg.experiment.randomseed)
        torch.cuda.manual_seed_all(cfg.experiment.randomseed)
    if configargs.cuda is not None:
        cfg.experiment.cuda = "cuda:" + configargs.cuda
    device = torch.device(cfg.experiment.cuda)

    log_path = os.path.expanduser(cfg.experiment.log_path)
    data_path = os.path.expanduser(cfg.dataset.data_path)

    if configargs.testing:
        writer = None
    else:
        writer = SummaryWriter(log_path)  # tensorboard --logdir=runs
        copyfile(__file__, os.path.join(log_path, 'train.py'))
        copyfile(configargs.config, os.path.join(log_path, 'config.yml'))

    start_epoch = cfg.experiment.start_epoch
    end_epoch = cfg.experiment.end_epoch
    batch_size = int(eval(cfg.experiment.batch_size))

    ##########################
    # Build data loader
    if 'pmsData' in data_path or 'DiLiGenT' in data_path:
        input_data_dict = load_diligent(data_path, cfg)
    elif 'LightStage' in data_path:
        input_data_dict = load_lightstage(data_path, scale=1)
    elif 'Apple' in data_path:
        input_data_dict = load_apple(data_path, scale=1)
    elif 'Sythn' in data_path:
        input_data_dict = load_sythn(cfg.dataset.syn_obj, cfg.dataset.light_index, cfg.dataset.material_idx)
    else:
        raise NotImplementedError('Unknown dataset')
    training_data_loader = Data_Loader(
        input_data_dict,
        gray_scale=cfg.dataset.gray_scale,
        data_len=300,
        mode='testing',
        shadow_threshold=cfg.dataset.shadow_threshold,
    )
    training_dataloader = torch.utils.data.DataLoader(training_data_loader, batch_size=batch_size, shuffle=not configargs.testing, num_workers=0)
    mask = training_data_loader.get_mask().to(device)
    idxp_invalid = np.where(mask.cpu().numpy() < 0.5)
    mean_xy = training_data_loader.get_mean_xy().to(device)
    bounding_box_xy = training_data_loader.get_bounding_box()[0].to(device), training_data_loader.get_bounding_box()[1].to(device)
    bounding_box_int = training_data_loader.get_bounding_box_int()
    unit_sphere_normal, unit_sphere_idxp, unit_sphere_invalididxp = training_data_loader.get_unitsphere_normal()
    unit_sphere_bounding_box_int = training_data_loader.get_unitsphere_bounding_box_int()
    all_light_direction = training_data_loader.get_all_light_direction().to(device)
    all_light_intensity = training_data_loader.get_all_light_intensity().to(device)
    images_max_value = training_data_loader.images_max.to(device)
    eval_data_len = len(training_data_loader) if configargs.testing else 1
    if configargs.quick_testing:
        eval_data_len = 1
        configargs.testing = True
    if cfg.experiment.eval_every_iter <= (end_epoch-start_epoch+1):
        eval_data_loader = Data_Loader(
            input_data_dict,
            gray_scale=cfg.dataset.gray_scale,
            data_len=eval_data_len,
            mode='testing',
            shadow_threshold=cfg.dataset.shadow_threshold,
        )
        eval_dataloader = torch.utils.data.DataLoader(eval_data_loader, batch_size=1, shuffle=False, num_workers=0)
    ##########################
    light_init = add_noise_light_init(
        all_light_direction.cpu(),
        all_light_intensity.cpu(),
        ld_noise=cfg.models.light_model.ld_noise,
        li_noise=cfg.models.light_model.li_noise
    )
    ##########################
    # Build model
    if cfg.models.use_mean_var:
        cfg.models.nerf.include_input_input2 += 2 if cfg.dataset.gray_scale else 6

    if cfg.models.specular.type == 'Spherical_Gaussian':
        NeRFModel_output_ch = cfg.models.specular.num_basis * (2 if cfg.dataset.gray_scale else 4)
    else:
        NeRFModel_output_ch = cfg.models.specular.num_basis

    model = NeRFModel_Separate(
        num_layers=cfg.models.nerf.num_layers,
        hidden_size=cfg.models.nerf.hidden_size,
        skip_connect_every=cfg.models.nerf.skip_connect_every,
        num_encoding_fn_input1=cfg.models.nerf.num_encoding_fn_input1,
        num_encoding_fn_input2=cfg.models.nerf.num_encoding_fn_input2,
        include_input_input1=cfg.models.nerf.include_input_input1,  # denote images coordinates (ix, iy)
        include_input_input2=cfg.models.nerf.include_input_input2,  # denote rgb latent code (lx, ly, lz)
        output_ch=NeRFModel_output_ch,
        gray_scale=cfg.dataset.gray_scale,
        mask=mask,
    )
    encode_fn_input1 = get_embedding_function(num_encoding_functions=cfg.models.nerf.num_encoding_fn_input1)
    model.train()
    model.to(device)

    if cfg.models.specular.type == 'Spherical_Gaussian':
        if hasattr(cfg.models.specular, 'trainable_k'):
            trainable_k = cfg.models.specular.trainable_k
        else:
            trainable_k = False
        specular_model = Spherical_Gaussian(
            num_basis=cfg.models.specular.num_basis,
            k_low=cfg.models.specular.k_low,
            k_high=cfg.models.specular.k_high,
            trainable_k=trainable_k,
        )
    specular_model.train()
    specular_model.to(device)

    if cfg.models.light_model.type == 'None':
        light_model = Light_Model(light_init=light_init, num_rays=np.count_nonzero(input_data_dict['mask']), requires_grad=True)
        light_model.to(device)
    elif cfg.models.light_model.type == 'Light_Model_CNN':
        light_model = Light_Model_CNN(
            num_layers=cfg.models.light_model.num_layers,
            hidden_size=cfg.models.light_model.hidden_size,
            output_ch=4,
            batchNorm=False
        )
        light_model.train()
        light_model.to(device)
    else:
        raise NotImplementedError('Unknown light model')

    if cfg.models.light_model.load_pretrain:
        model_checkpoint_pth = os.path.expanduser(cfg.models.light_model.load_pretrain)
        if model_checkpoint_pth[-4:] != '.pth':
            model_checkpoint_pth = sorted(glob.glob(os.path.join(model_checkpoint_pth, 'model*.pth')))[-1]
        print('Found pretrain light model checkpoints: ', model_checkpoint_pth)
        ckpt = torch.load(model_checkpoint_pth, map_location=device)
        light_model.load_state_dict(ckpt['model_state_dict'])
        light_model.set_images(
            num_rays=np.count_nonzero(input_data_dict['mask']),
            images=training_data_loader.get_all_masked_images(),
            device=device,
        )
        light_model.init_explicit_lights(
            explicit_direction=cfg.models.light_model.explicit_direction,
            explicit_intensity=cfg.models.light_model.explicit_intensity,
        )

    params_list = []
    params_list.append({'params': model.parameters()})

    params_list.append({'params': specular_model.parameters()})

    if hasattr(cfg.optimizer, 'light_lr') and cfg.optimizer.light_lr is not None:
        params_list.append({'params': light_model.parameters(), 'lr': cfg.optimizer.light_lr})
    else:
        params_list.append({'params': light_model.parameters()})

    optimizer = optim.Adam(params_list, lr=cfg.optimizer.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler.step_size, gamma=cfg.scheduler.gamma)
    ##########################

    ##########################
    # Load checkpoints
    if configargs.testing:
        cfg.models.load_checkpoint = True
        cfg.models.checkpoint_path = log_path
    if cfg.models.load_checkpoint:
        model_checkpoint_pth = os.path.expanduser(cfg.models.checkpoint_path)
        if model_checkpoint_pth[-4:] != '.pth':
            model_checkpoint_pth = sorted(glob.glob(os.path.join(model_checkpoint_pth, 'model*.pth')))[-1]
        print('Found checkpoints', model_checkpoint_pth)
        ckpt = torch.load(model_checkpoint_pth, map_location=device)

        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        specular_model.load_state_dict(ckpt['specular_model_state_dict'])

        light_model.load_state_dict(ckpt['light_model_state_dict'])
        start_epoch = ckpt['global_step']+1
    if configargs.testing:
        start_epoch = 1
        end_epoch = 1
        cfg.experiment.eval_every_iter = 1
        cfg.experiment.save_every_iter = 100
    if configargs.quick_testing:
        cfg.experiment.eval_every_iter = 100000000
    ##########################

    if cfg.loss.rgb_loss == 'l1':
        rgb_loss_function = F.l1_loss
    elif cfg.loss.rgb_loss == 'l2':
        rgb_loss_function = F.mse_loss
    else:
        raise AttributeError('Undefined rgb loss function.')

    start_t = time.time()
    h, w = mask.size(0), mask.size(1)
    idxp = torch.where(mask > 0.5)
    num_rays = len(idxp[0])
    iters_per_epoch = len(training_dataloader)

    #####
    epoch = 0
    model.eval()
    with torch.no_grad():
        print('================ evaluation results===============')
        for eval_idx, eval_datain in enumerate(eval_dataloader, start=1):
            batch_size = 1
            train(input_data=eval_datain, testing=True)
        print('==================================================')
    model.train()
    #####

    for epoch in range(start_epoch, end_epoch+1):
        for iter_num, input_data in enumerate(training_dataloader):
            if not configargs.testing:
                batch_size = int(eval(cfg.experiment.batch_size))
                output_normal_0 = train(input_data=input_data, testing=False)

        scheduler.step()

        if epoch % cfg.experiment.save_every_epoch == 0:
            savepath = os.path.join(log_path, 'model_params_%05d.pth' % epoch)
            torch.save({
                'global_step': epoch,
                'model_state_dict': model.state_dict(),
                'specular_model_state_dict': specular_model.state_dict(),
                'light_model_state_dict': light_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, savepath)
            print('Saved checkpoints at', savepath)

        if epoch % cfg.experiment.eval_every_iter == 0:
            model.eval()
            with torch.no_grad():
                print('================ evaluation results===============')
                for eval_idx, eval_datain in enumerate(eval_dataloader, start=1):
                    batch_size = 1
                    train(input_data=eval_datain, testing=True)
                print('==================================================')
            model.train()
