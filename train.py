import torch
from torch.utils.data import DataLoader
from dataset import TrainDataset
from darknet import DarkNet
import cfgs.cfg_train as cfg
from tensorboardX import SummaryWriter
import time
import os
import shutil
import math
from util import Avger

if (not cfg.continue_train) and os.path.exists(cfg.model_dir):
    shutil.rmtree(cfg.model_dir)
os.makedirs(os.path.join(cfg.model_dir, "log"), exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(cfg.model_dir, "log"))
model = DarkNet(cfg)
if cfg.continue_train:
    model.load_state_dict(torch.load(os.path.join(
        cfg.model_dir, f"model_{cfg.last_epoch}.pkl")))
    log_file = open(os.path.join(cfg.model_dir, "log", "train.log"), "a")
else:
    if isinstance(model.MetaLearner, torch.nn.DataParallel):
        model.MetaLearner.module.load_from_origin_file(cfg.pretrain_darknet)
    else:
        model.MetaLearner.load_from_origin_file(cfg.pretrain_darknet)
    log_file = open(os.path.join(cfg.model_dir, "log", "train.log"), "w")
model.train()
dataset = TrainDataset(cfg.dataroot, cfg)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16)

factor = cfg.neg_lr_scale if cfg.dynamic else 1

print("learning rate:", cfg.init_learning_rate)

optim = torch.optim.SGD([{"params": model.parameters(), "initial_lr": cfg.init_learning_rate / factor}], lr=cfg.init_learning_rate / factor,
                        momentum=cfg.momentum, weight_decay=cfg.weight_decay)
if cfg.continue_train and cfg.last_epoch != -1:
    optim.load_state_dict(torch.load(os.path.join(
        cfg.model_dir, f"optim_{cfg.last_epoch}.pkl")))


def lr_func(ep):
    ret = 1
    for i, e in enumerate(cfg.lr_decay_epochs):
        if ep >= e:
            ret *= cfg.lr_decay[i]
    return ret


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optim, lr_func, last_epoch=cfg.last_epoch)
n_iter = (cfg.last_epoch+1) * len(dataset) * cfg.batch_size
model.seen = n_iter

if cfg.fine_tune:
    model.seen = 4000000

s = time.time()
lpt = 0
loss_cls = Avger()
loss_coord = Avger()
loss_obj = Avger()
acc = Avger()
for epoch in range(cfg.last_epoch+1, cfg.max_epoch):
    for data in dataloader:
        optim.zero_grad()
        # torch.save(data, "input")
        img = data["input"][0].to(model.device)
        box = list(map(lambda x: x[0].to(model.device), data["box"]))
        clas = list(map(lambda x: x[0].to(model.device), data["class"]))
        supp = data["supp"].to(model.device)
        l_cls, l_coord, l_obj, tacc = model.optimize(
            img, box, clas, supp)
        optim.step()
        loss_cls.step(l_cls)
        loss_coord.step(l_coord)
        loss_obj.step(l_obj)
        acc.step(tacc)
        writer.add_scalar("loss_cls", l_cls, global_step=n_iter)
        writer.add_scalar("loss_coord", l_coord, global_step=n_iter)
        writer.add_scalar("loss_obj", l_obj, global_step=n_iter)
        writer.add_scalar("acc", tacc, global_step=n_iter)
        n_iter += cfg.batch_size
        if n_iter // cfg.print_freq > lpt // cfg.print_freq:
            t = time.time()
            logs = f"it {n_iter} epoch {epoch} iter {n_iter-epoch*len(dataset)*cfg.batch_size} {(t-s)/(n_iter-lpt)}s per img, lr = {optim.param_groups[0]['lr']}, loss_cls: {loss_cls.get()}, loss_coord: {loss_coord.get()}, loss_obj: {loss_obj.get()}, acc: {acc.get()}"
            print(logs)
            log_file.write(logs+'\n')
            log_file.flush()
            if math.isinf(loss_cls.get()+loss_coord.get()+loss_obj.get()) or math.isnan(loss_cls.get()+loss_coord.get()+loss_obj.get()):
                input("error")
            loss_cls.reset()
            loss_coord.reset()
            loss_obj.reset()
            acc.reset()
            s = t
            lpt = n_iter
        # exit(0)
    scheduler.step()
    if (epoch+1) % cfg.save_freq == 0:
        torch.save(model.state_dict(), os.path.join(
            cfg.model_dir, f"model_{epoch}.pkl"))
        torch.save(optim.state_dict(), os.path.join(
            cfg.model_dir, f"optim_{epoch}.pkl"))
