import torch.nn as nn
import torch
import networks
import numpy as np


class DynamicNet(nn.Module):
    def __init__(self, cfg):
        super(DynamicNet, self).__init__()
        self.cfg = cfg
        self.device = torch.device("cuda:0") if cfg.gpu else torch.device("cpu")
        net_cfgs = [
            # conv1s
            [(32, 3)],
            ['M', (64, 3)],
            ['M', (128, 3), (64, 1), (128, 3)],
            ['M', (256, 3), (128, 1), (256, 3)],
            ['M', (512, 3), (256, 1), (512, 3), (256, 1), (512, 3)],
            # conv2
            ['M', (1024, 3), (512, 1), (1024, 3), (512, 1), (1024, 3)],
            # ------------
            # conv3
            [(1024, 3), (1024, 3)],
            # conv4
            [(1024, 3)]
        ]

        # darknet
        self.conv1s, c1 = networks._make_layers(3, net_cfgs[0:5])
        self.conv2, c2 = networks._make_layers(c1, net_cfgs[5])
        # ---
        self.conv3, c3 = networks._make_layers(c2, net_cfgs[6])

        stride = 2
        # stride*stride times the channels of conv1s
        self.reorg = networks.ReorgLayer(stride=2)
        # cat [conv1s, conv3]
        self.conv4, c4 = networks._make_layers((c1*(stride*stride) + c3), net_cfgs[7])

        # linear
        if self.cfg.dynamic:
            out_channels = cfg.num_anchors * 6
        else:
            out_channels = cfg.num_anchors * (14+6)
        self.conv5 = networks.Conv2d(c4, out_channels, 1, 1, relu=False)
        self.to(self.device)
    
    def forward(self, img, dynamic_weights=None):
        # dynamic_weights: b * num_cls * channel
        conv1s = self.conv1s(img)
        conv2 = self.conv2(conv1s)
        conv3 = self.conv3(conv2)
        reorg = self.reorg(conv1s)
        fuse = torch.cat((reorg, conv3), dim=1)
        conv4 = self.conv4(fuse)
        b, c, h, w = conv4.shape
        
        if self.cfg.dynamic and dynamic_weights is not None:
            num_cls = dynamic_weights.size(1)
            reweighting = conv4.unsqueeze(1) # b * num_cls(1) * c * h * w
            # print("output_feature", reweighting.shape)
            # print("dynamic", dynamic_weights.unsqueeze(-1).unsqueeze(-1).shape)
            # print(dynamic_weights[0,:,:10])
            reweighting = reweighting * dynamic_weights.unsqueeze(-1).unsqueeze(-1)
            reweighting = reweighting.reshape(b*num_cls, c, h, w)
            pred = self.conv5(reweighting)
            pred = pred.reshape(b*num_cls, self.cfg.num_anchors, 6, h, w).permute(0,3,4,1,2).contiguous()
            # b.num_cls * h * w * num_anchors * 6
            xy_pred = torch.sigmoid(pred[:,:,:,:,:2])
            wh_pred = pred[:,:,:,:,2:4]
            iou_pred = torch.sigmoid(pred[:,:,:,:,4])
            cls_pred = pred[:,:,:,:,5]
        else:
            pred = self.conv5(conv4)
            pred = pred.reshape(b, self.cfg.num_anchors, 20, h, w).permute(0,3,4,1,2).contiguous()
            xy_pred = torch.sigmoid(pred[:,:,:,:,:2])
            wh_pred = pred[:,:,:,:,2:4]
            iou_pred = torch.sigmoid(pred[:,:,:,:,4])
            cls_pred = pred[:,:,:,:,5:]
        return xy_pred, wh_pred, iou_pred, cls_pred

    def load_from_npz(self, fname, num_conv=18):
        dest_src = {'conv.weight': 'kernel', 'conv.bias': 'biases',
                    'bn.weight': 'gamma', 'bn.bias': 'biases',
                    'bn.running_mean': 'moving_mean',
                    'bn.running_var': 'moving_variance'}
        params = np.load(fname)
        own_dict = self.state_dict()
        keys = list(filter(lambda x: x.find("num_batches_tracked") < 0, list(own_dict.keys())))

        for i, start in enumerate(range(0, len(keys), 5)):
            if num_conv is not None and i >= num_conv:
                break
            end = min(start+5, len(keys))
            for key in keys[start:end]:
                list_key = key.split('.')
                ptype = dest_src['{}.{}'.format(list_key[-2], list_key[-1])]
                src_key = '{}-convolutional/{}:0'.format(i, ptype)
                print((src_key, own_dict[key].size(), params[src_key].shape))
                param = torch.from_numpy(params[src_key])
                if ptype == 'kernel':
                    param = param.permute(3, 2, 0, 1)
                own_dict[key].copy_(param)
    
    def load_from_origin_file(self, weightfile):
        fp = open(weightfile, 'rb')
        _ = np.fromfile(fp, count=4, dtype=np.int32)
        buf = np.fromfile(fp, dtype = np.float32)
        fp.close()

        start = 0
        start = load_net(self.conv1s, buf, start)
        start = load_net(self.conv2, buf, start)
        print("load finish:", start, len(buf))

def load_net(seq, buf, start):
    for model in seq:
        if isinstance(model, networks.Conv2d_BatchNorm):
            start = model.load_(buf, start)
        elif isinstance(model, nn.Sequential):
            start = load_net(model, buf, start)
    return start