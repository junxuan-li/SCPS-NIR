# parts of the code are borrow from https://github.com/guanyingc/SDPS-Net
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_


def conv(batchNorm, cin, cout, k=3, stride=1, pad=-1):
    pad = pad if pad >= 0 else (k - 1) // 2
    if batchNorm:
        print('=> convolutional layer with bachnorm')
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.1, inplace=True)
                )
    else:
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True),
                nn.LeakyReLU(0.1, inplace=True)
                )


def outputConv(cin, cout, k=3, stride=1, pad=1):
    return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True))


# Classification
class FeatExtractor(nn.Module):
    def __init__(self, batchNorm, c_in, c_out=256):
        super(FeatExtractor, self).__init__()
        self.conv1 = conv(batchNorm, c_in, 64, k=3, stride=2, pad=1)
        self.conv2 = conv(batchNorm, 64, 128, k=3, stride=2, pad=1)
        self.conv3 = conv(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.conv4 = conv(batchNorm, 128, 128, k=3, stride=2, pad=1)
        self.conv5 = conv(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.conv6 = conv(batchNorm, 128, 256, k=3, stride=2, pad=1)
        self.conv7 = conv(batchNorm, 256, 256, k=3, stride=1, pad=1)

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        return out


class Classifier(nn.Module):
    def __init__(self, batchNorm, c_in):
        super(Classifier, self).__init__()
        self.conv1 = conv(batchNorm, 256, 256, k=3, stride=1, pad=1)
        self.conv2 = conv(batchNorm, 256, 256, k=3, stride=2, pad=1)
        self.conv3 = conv(batchNorm, 256, 256, k=3, stride=2, pad=1)
        self.conv4 = conv(batchNorm, 256, 256, k=3, stride=2, pad=1)

        self.dir_x_est = nn.Sequential(
            conv(batchNorm, 256, 64, k=1, stride=1, pad=0),
            outputConv(64, 1, k=1, stride=1, pad=0))
        self.dir_y_est = nn.Sequential(
            conv(batchNorm, 256, 64, k=1, stride=1, pad=0),
            outputConv(64, 1, k=1, stride=1, pad=0))
        self.dir_z_est = nn.Sequential(
            conv(batchNorm, 256, 64, k=1, stride=1, pad=0),
            outputConv(64, 1, k=1, stride=1, pad=0))

        self.int_est = nn.Sequential(
            conv(batchNorm, 256, 64, k=1, stride=1, pad=0),
            outputConv(64, 1, k=1, stride=1, pad=0))

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        dir_x = self.dir_x_est(out)
        dir_y = self.dir_y_est(out)
        dir_z = -torch.abs(self.dir_z_est(out))
        dir_est = torch.cat([dir_x, dir_y, dir_z], dim=1)

        outputs = {}
        outputs['dirs'] = nn.functional.normalize(dir_est, p=2, dim=1)
        outputs['ints'] = torch.abs(self.int_est(out))
        return outputs


class LNet(nn.Module):
    def __init__(self, batchNorm=False, c_in=3):
        super(LNet, self).__init__()
        self.featExtractor = FeatExtractor(batchNorm, c_in, 128)
        self.classifier = Classifier(batchNorm, 256)
        self.c_in = c_in

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs, idx=None):
        if inputs is None:
            x = self.images[idx].to(self.device)
        else:
            x = inputs

        net_input = self.featExtractor(x)
        outputs = self.classifier(net_input)

        if inputs is not None:
            return outputs
        else:
            num_rays = self.num_rays
            out_ld, out_li = outputs['dirs'].squeeze(), outputs['ints'].squeeze()
            out_ld_r = out_ld[:, None, :].repeat(1, num_rays, 1)  # (96, num_rays, 3)
            out_ld_r = out_ld_r.view(-1, 3)  # (96*num_rays, 3)

            out_li_r = out_li[:, None, None].repeat(1, num_rays, 3)
            out_li_r = out_li_r.view(-1, 3)  # (96*num_rays, 1)
            return out_ld_r, out_li_r

    def set_images(self, num_rays, images, device):
        self.num_rays = num_rays
        self.images = images
        self.device = device
        return

    def get_all_lights(self, device):
        inputs = self.images.to(self.device)

        net_input = self.featExtractor(inputs)
        outputs = self.classifier(net_input)

        out_ld, out_li = outputs['dirs'].squeeze(), outputs['ints'].squeeze()[:, None]

        return out_ld, out_li.repeat(1, 3)
