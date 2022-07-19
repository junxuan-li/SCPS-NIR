import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Model
class Light_Model(nn.Module):
    def __init__(self, num_rays, light_init, requires_grad=True):
        super(Light_Model, self).__init__()
        light_direction_xy = light_init[0][:, :-1].clone().detach()
        light_direction_z = light_init[0][:, -1:].clone().detach()
        light_intensity = light_init[1].mean(dim=-1, keepdims=True).clone().detach()

        self.light_direction_xy = nn.Parameter(light_direction_xy.float(), requires_grad=requires_grad)
        self.light_direction_z = nn.Parameter(light_direction_z.float(), requires_grad=requires_grad)
        self.light_intensity = nn.Parameter(light_intensity.float(), requires_grad=requires_grad)

        self.num_rays = num_rays

    def forward(self, idx):
        num_rays = self.num_rays
        out_ld = torch.cat([self.light_direction_xy[idx], -torch.abs(self.light_direction_z[idx])], dim=-1)
        out_ld = F.normalize(out_ld, p=2, dim=-1)[:, None, :]  # (96, 1, 3)

        out_ld = out_ld.repeat(1, num_rays, 1)
        out_ld = out_ld.view(-1, 3)  # (96*num_rays, 3)

        out_li = torch.abs(self.light_intensity[idx])[:, None, :]  # (96, 1, 1)
        out_li = out_li.repeat(1, num_rays, 3)
        out_li = out_li.view(-1, 3)  # (96*num_rays, 3)
        return out_ld, out_li

    def get_light_from_idx(self, idx):
        out_ld_r, out_li_r = self.forward(idx)
        return out_ld_r, out_li_r

    def get_all_lights(self):
        with torch.no_grad():
            light_direction_xy = self.light_direction_xy
            light_direction_z = -torch.abs(self.light_direction_z)
            light_intensity = torch.abs(self.light_intensity).repeat(1, 3)

            out_ld = torch.cat([light_direction_xy, light_direction_z], dim=-1)
            out_ld = F.normalize(out_ld, p=2, dim=-1)  # (96, 3)
            return out_ld, light_intensity


class Light_Model_CNN(nn.Module):
    def __init__(
            self,
            num_layers=3,
            hidden_size=64,
            output_ch=4,
            batchNorm=False
    ):
        super(Light_Model_CNN, self).__init__()
        self.conv1 = conv_layer(batchNorm, 4, 64,  k=3, stride=2, pad=1, afunc='LReLU')
        self.conv2 = conv_layer(batchNorm, 64, 128,  k=3, stride=2, pad=1)
        self.conv3 = conv_layer(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.conv4 = conv_layer(batchNorm, 128, 128,  k=3, stride=2, pad=1)
        self.conv5 = conv_layer(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.conv6 = conv_layer(batchNorm, 128, 256,  k=3, stride=2, pad=1)
        self.conv7 = conv_layer(batchNorm, 256, 256,  k=3, stride=1, pad=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = torch.nn.functional.relu
        self.dir_linears = nn.ModuleList(
            [nn.Linear(256, hidden_size)] + [nn.Linear(hidden_size, hidden_size) for i in range(num_layers - 1)])
        self.output_linear = nn.Linear(hidden_size, output_ch)

    def forward(self, inputs):
        x = inputs
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        for i, l in enumerate(self.dir_linears):
            out = self.dir_linears[i](out)
            out = self.relu(out)
        outputs = self.output_linear(out)

        light_direction_xy = outputs[:, :2]
        light_direction_z = -torch.abs(outputs[:, 2:3])-0.1
        light_intensity = torch.abs(outputs[:, 3:])

        out_ld = torch.cat([light_direction_xy, light_direction_z], dim=-1)
        out_ld = F.normalize(out_ld, p=2, dim=-1)  # (96, 3)
        out_li = light_intensity  # (96, 1)

        outputs = {}
        outputs['dirs'] = out_ld
        outputs['ints'] = out_li
        return outputs

    def set_images(self, num_rays, images, device):
        self.num_rays = num_rays
        self.images = images
        self.device = device
        return

    def get_light_from_idx(self, idx):
        if hasattr(self, 'explicit_model'):
            out_ld_r, out_li_r = self.explicit_model(idx)
        else:
            x = self.images[idx].to(self.device)
            outputs = self.forward(x)
            out_ld, out_li = outputs['dirs'], outputs['ints'].repeat(1, 3)

            num_rays = self.num_rays
            out_ld_r = out_ld[:, None, :].repeat(1, num_rays, 1)  # (96, num_rays, 3)
            out_ld_r = out_ld_r.view(-1, 3)  # (96*num_rays, 3)

            out_li_r = out_li[:, None, :].repeat(1, num_rays, 1)
            out_li_r = out_li_r.view(-1, 3)  # (96*num_rays, 3)
        return out_ld_r, out_li_r

    def get_all_lights(self):
        if hasattr(self, 'explicit_model'):
            out_ld, out_li = self.explicit_model.get_all_lights()
        else:
            inputs = self.images.to(self.device)
            outputs = self.forward(inputs)
            out_ld, out_li = outputs['dirs'], outputs['ints']
        return out_ld, out_li.repeat(1,3)

    def init_explicit_lights(self, explicit_direction=False, explicit_intensity=False):
        if explicit_direction or explicit_intensity:
            light_init = self.get_all_lights()
            self.explicit_intensity = explicit_intensity
            self.explicit_direction = explicit_direction
            self.explicit_model = Light_Model(self.num_rays, light_init, requires_grad=True)
        else:
            return


def activation(afunc='LReLU'):
    if afunc == 'LReLU':
        return nn.LeakyReLU(0.1, inplace=True)
    elif afunc == 'ReLU':
        return nn.ReLU(inplace=True)
    else:
        raise Exception('Unknown activation function')

def conv_layer(batchNorm, cin, cout, k=3, stride=1, pad=-1, afunc='LReLU'):
    if type(pad) != tuple:
        pad = pad if pad >= 0 else (k - 1) // 2
    mList = [nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True)]
    if batchNorm:
        print('=> convolutional layer with batchnorm')
        mList.append(nn.BatchNorm2d(cout))
    mList.append(activation(afunc))
    return nn.Sequential(*mList)


class NeRFModel_Separate(torch.nn.Module):
    def __init__(
        self,
        num_layers=8,
        hidden_size=256,
        skip_connect_every=3,
        num_encoding_fn_input1=10,
        num_encoding_fn_input2=0,
        include_input_input1=2,   # denote images coordinates (ix, iy)
        include_input_input2=0,   # denote lighting direcions (lx, ly, lz)
        output_ch=1,
        gray_scale=False,
        mask=None,
    ):
        super(NeRFModel_Separate, self).__init__()
        self.dim_ldir = include_input_input2 * (1 + 2 * num_encoding_fn_input2)
        self.dim_ixiy = include_input_input1 * (1 + 2 * num_encoding_fn_input1) + self.dim_ldir
        self.dim_ldir = 0
        self.skip_connect_every = skip_connect_every + 1

        ##### Layers for Material Map #####
        self.layers_xyz = torch.nn.ModuleList()
        self.layers_xyz.append(torch.nn.Linear(self.dim_ixiy, hidden_size))
        for i in range(1, num_layers):
            if i == self.skip_connect_every:
                self.layers_xyz.append(torch.nn.Linear(self.dim_ixiy + hidden_size, hidden_size))
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))
        ###################################

        ##### Layers for Normal Map #####
        self.layers_xyz_normal = torch.nn.ModuleList()
        self.layers_xyz_normal.append(torch.nn.Linear(self.dim_ixiy, hidden_size))
        for i in range(1, num_layers):
            if i == self.skip_connect_every:
                self.layers_xyz_normal.append(torch.nn.Linear(self.dim_ixiy + hidden_size, hidden_size))
            else:
                self.layers_xyz_normal.append(torch.nn.Linear(hidden_size, hidden_size))
        ###################################

        # self.relu = torch.nn.functional.leaky_relu
        self.relu = torch.nn.functional.relu
        self.mask = mask
        self.idxp = torch.where(self.mask > 0.5)

        self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(hidden_size + self.dim_ldir, hidden_size // 2))
        for i in range(3):
            self.layers_dir.append(torch.nn.Linear(hidden_size // 2, hidden_size // 2))
        self.fc_spec_coeff = torch.nn.Linear(hidden_size//2,  output_ch)

        self.fc_diff = torch.nn.Linear(hidden_size // 2, 1 if gray_scale else 3)

        self.fc_normal_xy = torch.nn.Linear(hidden_size, 2)
        self.fc_normal_z = torch.nn.Linear(hidden_size, 1)

    def forward(self, input):
        xyz = input[..., : self.dim_ixiy]

        ##### Compute Normal Map #####
        x = xyz
        for i in range(len(self.layers_xyz_normal)):
            if i == self.skip_connect_every:
                x = self.layers_xyz_normal[i](torch.cat((xyz, x), -1))
            else:
                x = self.layers_xyz_normal[i](x)
            x = self.relu(x)
        normal_xy = self.fc_normal_xy(x)
        normal_z = -torch.abs(self.fc_normal_z(x))  # n_z is always facing camera
        normal = torch.cat([normal_xy, normal_z], dim=-1)
        normal = F.normalize(normal, p=2, dim=-1)
        ###################################

        ##### Compute Materaial Map #####
        x = xyz
        for i in range(len(self.layers_xyz)):
            if i == self.skip_connect_every:
                x = self.layers_xyz[i](torch.cat((xyz, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        if self.dim_ldir > 0:
            light_xyz = input[..., -self.dim_ldir:]
            feat = torch.cat([feat, light_xyz], dim=-1)
        x = self.layers_dir[0](feat)
        x = self.relu(x)
        for i in range(1, len(self.layers_dir)):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        spec_coeff = self.fc_spec_coeff(x)
        diff = torch.abs(self.fc_diff(x))
        ###################################
        return normal, diff, spec_coeff


class Spherical_Gaussian(nn.Module):
    def __init__(
            self,
            num_basis,
            k_low,
            k_high,
            trainable_k,
    ):
        super(Spherical_Gaussian, self).__init__()
        self.num_basis = num_basis

        self.trainable_k = trainable_k
        if self.trainable_k:
            kh = math.log10(k_high)
            kl = math.log10(k_low)
            self.k = nn.Parameter(torch.linspace(kh, kl, num_basis, dtype=torch.float32)[None, :])  # (1, num_basis)
        else:
            kh = math.log10(k_high)
            kl = math.log10(k_low)
            self.k = torch.linspace(kh, kl, num_basis, dtype=torch.float32)[None, :]

    def forward(self, light, normal, mu, view=None):
        if view is None:
            view = torch.zeros_like(light)
            view[..., 2] = -1
            view = view.detach()
        light, view, normal = F.normalize(light, p=2, dim=-1), F.normalize(view, p=2, dim=-1), F.normalize(normal, p=2, dim=-1)
        H = F.normalize((view + light) / 2, p=2, dim=-1)
        if self.trainable_k:
            k = self.k
        else:
            k = self.k.to(light.device)
        rate = 10 ** k  # range: 1 ~ 1000
        out = torch.abs(mu) * torch.exp(rate * ((H * normal).sum(dim=-1, keepdim=True) - 1))[..., None]  # (batch, num_basis, 3)
        return out


def Fresnel_Factor(light, half, view, normal):
    c = torch.abs((light * half).sum(dim=-1))
    g = torch.sqrt(1.33**2 + c**2 - 1)
    temp = (c*(g+c)-1)**2 / (c*(g-c)+1)**2
    f = (g-c)**2 / (2*(g+c)**2) * (1 + temp)
    return f


def totalVariation(image, mask, num_rays):
    pixel_dif1 = torch.abs(image[1:, :, :] - image[:-1, :, :]).sum(dim=-1) * mask[1:, :] * mask[:-1, :]
    pixel_dif2 = torch.abs(image[:, 1:, :] - image[:, :-1, :]).sum(dim=-1) * mask[:, 1:] * mask[:, :-1]
    tot_var = (pixel_dif1.sum() + pixel_dif2.sum()) / num_rays
    return tot_var


def totalVariation_L2(image, mask, num_rays):
    pixel_dif1 = torch.square(image[1:, :, :] - image[:-1, :, :]).sum(dim=-1) * mask[1:, :] * mask[:-1, :]
    pixel_dif2 = torch.square(image[:, 1:, :] - image[:, :-1, :]).sum(dim=-1) * mask[:, 1:] * mask[:, :-1]
    tot_var = (pixel_dif1.sum() + pixel_dif2.sum()) / num_rays
    return tot_var
