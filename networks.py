import torch
import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, same_padding=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding=padding)
        self.relu = nn.LeakyReLU(0.1, inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Conv2d_BatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, same_padding=False):
        super(Conv2d_BatchNorm, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01)
        self.relu = nn.LeakyReLU(0.1, inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
    def load_(self, buf, start):
        num_w = self.conv.weight.numel()
        num_b = self.bn.bias.numel()
        print(self.bn, num_b)
        print(self.conv, num_w)
        self.bn.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]))
        start = start + num_b
        self.bn.weight.data.copy_(torch.from_numpy(buf[start:start+num_b]))
        start = start + num_b
        self.bn.running_mean.copy_(torch.from_numpy(buf[start:start+num_b]))
        start = start + num_b
        self.bn.running_var.copy_(torch.from_numpy(buf[start:start+num_b]))
        start = start + num_b
        self.conv.weight.data.copy_(torch.from_numpy(buf[start:start+num_w]).view(self.conv.weight.data.shape))
        start = start + num_w
        return start


class ReorgLayer(nn.Module):
    def __init__(self, stride=1):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        b, c, h, w = x.shape
        outh = h//self.stride
        outw = w//self.stride
        outc = c*self.stride*self.stride
        x = x.reshape(b, c, outh, self.stride, outw,
                      self.stride).transpose(3, 4).contiguous()
        x = x.reshape(b, c, outh, outw, self.stride *
                      self.stride).permute(0, 1, 4, 2, 3).contiguous()
        return x.reshape(b, outc, outh, outw)


def _make_layers(in_channels, net_cfg):
    layers = []
    print(net_cfg)
    if len(net_cfg) > 0 and isinstance(net_cfg[0], list):
        for sub_cfg in net_cfg:
            layer, in_channels = _make_layers(in_channels, sub_cfg)
            layers.append(layer)
    else:
        for item in net_cfg:
            if item == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels, ksize = item
                layers.append(Conv2d_BatchNorm(in_channels,
                                               out_channels,
                                               ksize,
                                               same_padding=True))
                # layers.append(net_utils.Conv2d(in_channels, out_channels,
                #     ksize, same_padding=True))
                in_channels = out_channels

    return nn.Sequential(*layers), in_channels
