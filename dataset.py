import torch
from torch.utils.data import Dataset, DataLoader
import os, json
import PIL.Image as Image
import numpy as np
import random
import torchvision.transforms as transforms
import math


def rand_scale(r):
    s = float(torch.rand((1,)))*(r-1)+1
    if int(torch.randint(0, 2, (1,))) == 0:
        return s
    return 1 / s

def transformdata(img, size=256, scale=0.2, hue=0.1, saturation=1.5, exposure=1.5, aug_flag=True):
    if isinstance(size, int):
        size = (size, size)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if aug_flag:
        w, h = size
        scale = float(torch.rand((1,))) * scale + 1
        size = (round(w*scale), round(h*scale))

        sw = size[0]/img.width
        sh = size[1]/img.height
        offx = int(torch.randint(0, size[0]-w+1, (1,)))
        offy = int(torch.randint(0, size[1]-h+1, (1,)))

        img = transforms.functional.resize(img, size[::-1])
        img = transforms.functional.crop(img, offy, offx, h, w)

        flip = int(torch.randint(0, 2, (1,)))
        if flip == 1:
            img = transforms.functional.hflip(img)

        hue = (float(torch.rand((1,)))*2-1)*hue
        saturation = rand_scale(saturation)
        exposure = rand_scale(exposure)
        
        img = transforms.functional.adjust_brightness(img, exposure)
        img = transforms.functional.adjust_saturation(img, saturation)
        img = transforms.functional.adjust_hue(img, hue)
        return transforms.functional.to_tensor(img), {"flip": flip, "sh": sh, "sw": sw, 'offx': offx, "offy": offy , "size":(w, h)}
    else:
        h, w = (img.height, img.width)
        img = transforms.functional.resize(img, size)
        return transforms.functional.to_tensor(img), {"flip": 0, "sh": size[0]/h, "sw": size[1]/w, "offx": 0, "offy": 0, "size":size}

def transformbox(box, seed):
    xmin, ymin, xmax, ymax = box
    ymin = ymin * seed["sh"] - seed["offy"]
    ymax = ymax * seed["sh"] - seed["offy"]
    xmin = xmin * seed["sw"] - seed["offx"]
    xmax = xmax * seed["sw"] - seed["offx"]
    if seed["flip"] == 1:
        xmin, xmax = (seed["size"][0]-1-xmax, seed["size"][0]-1-xmin)
    xmin = min(seed["size"][0]-1., max(xmin, 0.))
    xmax = min(seed["size"][0]-1., max(xmax, 0.))
    ymin = min(seed["size"][1]-1., max(ymin, 0.))
    ymax = min(seed["size"][1]-1., max(ymax, 0.))
    if ymax-ymin < 1e-3 or xmax - xmin < 1e-3:
        return None
    return (xmin, ymin, xmax, ymax)

def genmask(size, box):
    # box: x, y
    ret = torch.zeros(size)
    ret[round(box[1]):round(box[3])+1,round(box[0]):round(box[2])+1] = 1
    return ret

class TrainDataset(Dataset):
    def __init__(self, dataroot, cfg):
        super(TrainDataset, self).__init__()
        self.fine_tune = cfg.fine_tune
        with open(os.path.join(dataroot, "fewshot_imgs.json"), "r") as f:
            self.few_imgs = json.load(f)
        with open(os.path.join(dataroot, "fewshot_label.json"), "r") as f:
            self.few_label = json.load(f)
        with open(os.path.join(dataroot, "img2box.json"), "r") as f:
            self.img2box = json.load(f)
        with open(os.path.join(dataroot, "label2class.json"), "r") as f:
            self.label2class = json.load(f)
        with open(os.path.join(dataroot, "label2imgs.json"), "r") as f:
            self.label2imgs = json.load(f)
        
        for k, v in enumerate(self.label2imgs):
            self.label2imgs[k] = map(lambda x: (tuple(map(lambda y:y-1, x[0])), x[1]), v)
        for k, v in self.img2box.items():
            self.img2box[k] = tuple(map(lambda x: (x[0], tuple(map(lambda y:y-1, x[1]))), v))

        if self.fine_tune:
            self.imgs = list(set(sum(self.few_imgs, [])))
            self.relabel = list(range(len(self.label2class)))
            self.select = self.make_selectdict(self.few_imgs)
        else:
            self.relabel = list()
            for i in range(len(self.label2class)):
                if i not in self.few_label:
                    self.relabel.append(i)

            self.imgs = list()
            for k, v in self.img2box.items():
                fl = False
                for box in v:
                    if box[0] in self.relabel:
                        fl = True
                        break
                if fl:
                    self.imgs.append(k)

            self.select = list()
            st = set(self.imgs)
            for label in self.relabel:
                tmp = list()
                for f in self.label2imgs[label]:
                    tmp.append(f)
                self.select.append(tmp)
            del st
        
        self.imgs = list(set(self.imgs))
        self.imgs.sort()
        self.device = torch.device("cuda:0") if cfg.gpu else torch.device("cpu")
        self.reweight_size = cfg.reweight_size
        self.multi_scale = getattr(cfg, "multi_scale_in", [416,])
        self.batch_size = cfg.batch_size

    def make_selectdict(self, ls):
        ret = [list() for _ in range(len(self.relabel))]
        for f in set(sum(ls, [])):
            boxes = self.img2box[f]
            for box in boxes:
                if f in self.few_imgs[box[0]]:
                    ret[box[0]].append((tuple(box[1]), f))
        ret = list(map(lambda x: list(set(x)), ret))
        return ret

    def __len__(self):
        return math.ceil(len(self.imgs) / self.batch_size)

    def get_img(self, index, size_ind):
        fname = self.imgs[index]
        input_img = Image.open(fname)
        input_img, seed = transformdata(input_img, size=self.multi_scale[size_ind])
        boxes = list()
        classes = list()
        debug = 0
        for box in self.img2box[fname]:
            assert(debug<=box[0])
            debug = box[0]
            # if box[0] != 8:
            #     continue
            if box[0] not in self.relabel:
                continue
            if self.fine_tune and fname not in self.few_imgs[box[0]]:
                continue
            tbx = transformbox(box[1], seed)
            if tbx is None:
                continue
            boxes.append(tbx)
            classes.append(self.relabel.index(box[0]))
        return input_img, torch.Tensor(boxes), torch.LongTensor(classes)

    def __getitem__(self, index):
        input_img = list()
        boxes = list()
        classes = list()
        size_ind = torch.randint(0, len(self.multi_scale), (1,)).item()
        for i in range(index*self.batch_size, min(len(self.imgs), (index+1)*self.batch_size)):
            ainput_img, aboxes, aclasses = self.get_img(i, size_ind)
            input_img.append(ainput_img)
            boxes.append(aboxes)
            classes.append(aclasses)
        input_img = torch.stack(input_img)

        few_imgs = list()
        for i in range(len(self.select)):
            ridx = torch.randint(0, len(self.select[i]), (1,)).item()
            box, rname = self.select[i][ridx]
            img = Image.open(rname)
            img, seed = transformdata(img, size=self.reweight_size, scale=0.0)
            box = transformbox(box, seed)
            mask = genmask(seed["size"], box)
            few_imgs.append(torch.cat((img, mask.unsqueeze(0)), dim=0))
        return {"input": input_img, "box": boxes, "class": classes, "supp": torch.stack(few_imgs)}
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    class testopt:
        def __init__(self):
            pass
    cfg = testopt()
    cfg.fine_tune = False
    cfg.reweight_size = 416
    cfg.gpu = False
    cfg.batch_size = 1
    dataset = TrainDataset("dataset", cfg)
    dataloader = DataLoader(dataset, 1, shuffle=True)
    print(dataset.__len__())
    for data in dataloader:
        idx = random.randint(0, cfg.batch_size-1)
        input_img = data["input"]
        boxes = data["box"]
        classes = data["class"]
        plt.cla()
        fig = plt.gcf()
        ax = plt.gca()
        ax.axis('off')
        print(input_img.shape)
        input_img = input_img[0][idx].permute((1,2,0)).numpy()
        ax.imshow(input_img)
        boxes=boxes[idx][0]
        classes=classes[idx][0]
        for box, clas in zip(boxes, classes):
            cname = dataset.label2class[dataset.relabel[clas.item()]]
            box = box.numpy().tolist()
            box = list(map(lambda x: round(float(x)), box))
            rect = mpatches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], color="red", fill=False)
            ax.add_patch(rect)
            ax.annotate(cname, (box[0], box[1]))
        plt.show()
        break
