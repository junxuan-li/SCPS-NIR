import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import glob
import matplotlib.pyplot as plt
import yaml
import argparse
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from cfgnode import CfgNode

from LNet_model import LNet
from models import Light_Model_CNN
from LNet_dataloader import customDataloader
from utils import cal_ints_acc, cal_dirs_acc


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
        default="configs/LNet/template.yml",
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

    configargs = parser.parse_args()

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
    if torch.cuda.device_count() == 0:
        device = torch.device("cpu")
    elif torch.cuda.device_count() == 1:
        device = torch.device("cuda:0")
    else:  # device count >= 3
        device = torch.device(cfg.experiment.cuda)

    log_path = os.path.expanduser(cfg.experiment.log_path)

    if configargs.testing:
        pass
    else:
        writer = SummaryWriter(log_path)  # tensorboard --logdir=runs
        copyfile(__file__, os.path.join(log_path, 'train.py'))
        copyfile(configargs.config, os.path.join(log_path, 'config.yml'))

    start_epoch = cfg.experiment.start_epoch
    end_epoch = cfg.experiment.end_epoch
    batch_size = int(eval(cfg.experiment.batch_size))

    ##########################
    # Build data loader
    data_path1 = os.path.expanduser(cfg.dataset.data_path1)
    data_path2 = os.path.expanduser(cfg.dataset.data_path2)
    train_loader, test_loader = customDataloader(
        data_path1,
        data_path2,
        batch=batch_size,
        val_batch=batch_size,
        workers=8
    )
    iters_per_epoch = len(train_loader)
    ##########################


    ##########################
    # Build model
    if cfg.models.type == 'LNet':
        model = LNet(batchNorm=cfg.models.batchNorm, c_in=4)
    elif cfg.models.type == 'Light_Model_CNN':
        model = Light_Model_CNN(
            num_layers=3,
            hidden_size=64,
            output_ch=4,
            batchNorm=cfg.models.batchNorm
        )
    else:
        raise NotImplementedError('Unknown light model:', cfg.models.type)
    model.train()
    model.to(device)

    params_list = []
    params_list.append({'params': model.parameters()})
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
        start_epoch = ckpt['global_step']+1
    if configargs.testing:
        start_epoch = 1
        end_epoch = 1
        cfg.experiment.eval_every_iter = 1
        cfg.experiment.save_every_iter = 100
    ##########################

    start_t = time.time()

    for epoch in range(start_epoch, end_epoch+1):
        for iter_num, input_data in enumerate(train_loader):
            network_input = torch.cat([input_data['img'], input_data['mask']], dim=1).to(device)
            gt_dirs = input_data['dirs'].to(device)
            gt_ints = input_data['ints'].to(device).mean(dim=1)

            output = model(network_input)
            dir_loss = (1 - (output['dirs'].squeeze() * gt_dirs.squeeze()).sum(dim=-1)).mean()
            int_loss = F.mse_loss(output['ints'].squeeze(), gt_ints.squeeze())
            loss = dir_loss + cfg.loss.ints_alpha * int_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log the running loss
            cost_t = time.time() - start_t
            est_time = cost_t / ((epoch - start_epoch) * iters_per_epoch + iter_num + 1) * (
                    (end_epoch - epoch) * iters_per_epoch + iters_per_epoch - iter_num - 1)
            if iter_num % cfg.experiment.print_every_iter == 0:
                print(
                    'epoch: %d,  iter: %2d/ %d, dir_loss: %.4f, int_loss: %.4f  cost_time: %d m %2d s,  est_time: %d m %2d s' %
                    (epoch, iter_num + 1, iters_per_epoch, dir_loss.item(), int_loss.item() , cost_t // 60, cost_t % 60,
                     est_time // 60, est_time % 60))
                writer.add_scalar('Train loss', loss.item(), (epoch - 1) * iters_per_epoch + iter_num)
                writer.add_scalar('Train dir loss', dir_loss.item(), (epoch - 1) * iters_per_epoch + iter_num)
                writer.add_scalar('Train int loss', int_loss.item(), (epoch - 1) * iters_per_epoch + iter_num)

        scheduler.step()

        if epoch % cfg.experiment.save_every_epoch == 0:
            savepath = os.path.join(log_path, 'model_params_%05d.pth' % epoch)
            torch.save({
                'global_step': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, savepath)
            print('Saved checkpoints at', savepath)

        if epoch % cfg.experiment.eval_every_epoch == 0:
            model.eval()
            dir_loss = 0
            int_loss = 0
            loss = 0
            with torch.no_grad():
                print('================ evaluation results===============')
                for iter_num, input_data in enumerate(test_loader):
                    network_input = torch.cat([input_data['img'], input_data['mask']], dim=1).to(device)
                    gt_dirs = input_data['dirs'].to(device)
                    gt_ints = input_data['ints'].to(device).mean(dim=1)

                    output = model(network_input)
                    dir_loss += ((1 - (output['dirs'].squeeze() * gt_dirs.squeeze()).sum(dim=-1)).mean()).item()
                    int_loss += (F.mse_loss(output['ints'].squeeze(), gt_ints.squeeze())).item()
                    loss += dir_loss + cfg.loss.ints_alpha * int_loss

                dir_loss /= len(test_loader)
                int_loss /= len(test_loader)
                loss /= len(test_loader)
                # log the running loss
                print(
                    'epoch: %d,  dir_loss: %.4f, int_loss: %.4f ' %
                    (epoch,  dir_loss, int_loss)
                )
                writer.add_scalar('Val loss', loss, epoch)
                writer.add_scalar('Val dir loss', dir_loss, epoch)
                writer.add_scalar('Val int loss', int_loss, epoch)

            print('==================================================')
            model.train()
