import torch.nn as nn
import torch
import networks


class ReweightNet(nn.Module):
    def __init__(self, cfg):
        super(ReweightNet, self).__init__()
        self.cfg = cfg
        self.device = torch.device("cuda:0") if cfg.gpu else torch.device("cpu")
        inputc = 4
        net_cfgs = [
            # conv1s
            [(32, 3)],
            ['M', (64, 3)],
            ['M', (128, 3)],
            ['M', (256, 3)],
            ['M', (512, 3)],
            ['M', (1024, 3)],
            ['M', (1024, 3)],
        ]
        self.global_pool = nn.AdaptiveMaxPool2d((1,1))
        self.model, outc = networks._make_layers(inputc, net_cfgs)
        self.model.to(self.device)
    
    def forward(self, inputx):
        # input:  b x ncls x c x h x w
        b, ncls, c, h, w = inputx.shape
        weight = self.model(inputx.reshape(b*ncls, c, h, w))
        weight = self.global_pool(weight).reshape(b, ncls, -1)
        return weight