import torch
from torch.utils.data import DataLoader
from dataset import transformdata, genmask, transformbox
from darknet import DarkNet
import cfgs.cfg_test as cfg
from tensorboardX import SummaryWriter
import time
import os
import json
import pickle
import PIL.Image as Image

model = DarkNet(cfg)
model.load_state_dict(torch.load(cfg.model_path))
model.eval()

with open(os.path.join("coco", "fewshot_imgs.json"), "r") as f:
    few_imgs = json.load(f)
    for i in range(len(few_imgs)):
        few_imgs[i] = list(set(few_imgs[i]))
with open(os.path.join("coco", "img2box.json"), "r") as f:
    img2box = json.load(f)
with open(os.path.join("coco", "label2class.json"), "r") as f:
    label2class = json.load(f)


detfiles = list(map(lambda c: open("tmp/{}.txt".format(c), "w"), label2class))

with torch.no_grad():
    reweight = list()
    for c in range(len(label2class)):
        supp = 0
        cnt = 0
        for p in few_imgs[c]:
            print(p)
            img = Image.open(p)
            boxes = img2box[p]
            img, seed = transformdata(
                img, size=cfg.reweight_size, aug_flag=False)
            for box in boxes:
                if box[0] == c:
                    box = transformbox(box[1], seed)
                    mask = genmask(seed["size"], box)
                    rv = torch.cat((img, mask.unsqueeze(0)), dim=0).unsqueeze(
                        0).unsqueeze(0).to(model.device)
                    supp = supp + model.Reweight(rv)
                    cnt += 1
                    del mask
                    del rv
            del img
        reweight.append(supp/cnt)
    reweight = torch.cat(reweight, dim=1)

    fns = os.listdir("coco/images/val2017")
    st = 0
    bs = 12
    tt = list()
    while st < len(fns):
        ed = min(st + bs, len(fns))
        tt.append(fns[st:ed])
        st = ed

    DT = list()
    for nfn in tt:
        imgs = list()
        seeds = list()
        for fn in nfn:
            img = Image.open(os.path.join("coco/images/val2017", fn))
            print(fn)
            img, seed = transformdata(img, cfg.image_size, aug_flag=False)
            img = img.unsqueeze(0).to(model.device)
            imgs.append(img)
            seeds.append(seed)
        res = model.pred_bbox(
            torch.cat(imgs, dim=0), reweight, cfg.iou_threshold, cfg.nms_threshold)
        cnt = 0
        for xmin, ymin, xmax, ymax, pred_cls, score in res:
            fn = nfn[cnt]
            seed = seeds[cnt]
            for i in range(xmin.shape[0]):
                c = int(pred_cls[i])
                log = "{} {} {} {} {} {}\n".format(os.path.splitext(fn)[0], float(
                    score[i]), max(float(xmin[i])/seed["sw"], 0), max(float(ymin[i])/seed["sh"], 0), min(float(xmax[i]), cfg.image_size)/seed["sw"], min(float(ymax[i]), cfg.image_size)/seed["sh"])
                detfiles[c].write(log)
            cnt += 1
            
