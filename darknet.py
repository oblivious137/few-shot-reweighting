import torch.nn as nn
import torch.nn.functional as F
import torch
import networks
from dynamicnet import DynamicNet
from reweightnet import ReweightNet
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


class DarkNet(nn.Module):
    def __init__(self, cfg):
        super(DarkNet, self).__init__()
        self.cfg = cfg
        self.device = torch.device(
            "cuda:0") if cfg.gpu else torch.device("cpu")
        self.d_cnt = 1
        if cfg.gpu and torch.cuda.device_count() > 1:
            self.d_cnt = torch.cuda.device_count()
            self.MetaLearner = nn.DataParallel(DynamicNet(cfg), list(range(self.d_cnt)))
            if self.cfg.dynamic:
                self.Reweight = nn.DataParallel(ReweightNet(cfg), list(range(self.d_cnt)))
        else:
            self.MetaLearner = DynamicNet(cfg)
            if self.cfg.dynamic:
                self.Reweight = ReweightNet(cfg)
        self.seen = 0

    def forward(self, x, supp=None):
        # if len(supp.shape) == 6:
        #     supp = supp.reshape(supp.size(
        #         0)*supp.size(1), supp.size(2), supp.size(3), supp.size(4), supp.size(5))
        if self.cfg.dynamic:
            dynamic_weight = self.Reweight(supp) # b.ncls.c
            dynamic_weight = dynamic_weight.unsqueeze(0)
            dynamic_weight = torch.mean(dynamic_weight, dim=1)
            return self.MetaLearner(x, dynamic_weight.repeat((self.d_cnt, 1, 1)))
        else:
            return self.MetaLearner(x)

    def gen_bcmask(self, classes, num_cls):
        mask = [0] * (len(classes) * num_cls)
        for b, clss in enumerate(classes):
            debug = 0
            for c in clss:
                assert debug <= c.item()
                debug = c.item()
                mask[c.item()+b*num_cls] = 1
        npos = sum(mask)
        ratio = self.cfg.neg * npos / (len(mask)-npos)
        presum = list()
        ps = 0
        for i in range(len(mask)):
            if float(torch.rand((1,))) < ratio:
                mask[i] = 1
            ps += mask[i]
            presum.append(ps)
        # torch.save((mask, presum), "tmp_data")
        # for b, clss in enumerate(classes):
        #     for c in clss:
        #         assert mask[int(c)+b*num_cls] == 1
        # mask, presum = torch.load("tmp_data")
        return (torch.LongTensor(mask)==1).requires_grad_(False).to(self.device), presum

    def optimize_dynamic(self, x, boxes, classes, supp):
        # x: b x c x h x w
        # boxes: b x [? x 4]    sorted by class
        # classes: b x [?]      sorted by class
        # supp: b x (small batch) x num_class x c x h x w
        xy_pred, wh_pred, iou_pred, cls_pred = self.forward(x, supp)
        # pred: b.num_cls * h * w * num_anchors * 2
        # iou_pred, cls_pred: # b.num_cls * h * w * num_anchors
        wh_pred = torch.exp(wh_pred)
        bs = x.size(0)
        ncls = supp.size(-4)
        ih, iw = x.shape[2:]
        oh, ow = iou_pred.shape[1:3]
        na = len(self.cfg.anchors)
        nbox = sum(map(len, classes))
        batch_class_mask, presum = self.gen_bcmask(classes, ncls)
        # print("select", presum[-1], "from", batch_class_mask.shape, "batch_class")
        nbc = presum[-1]

        anchor_shape = torch.Tensor(self.cfg.anchors).to(self.device)
        squeezed_box = torch.cat(boxes, dim=0).detach()

        box_dict = scatter_by_batch_class(classes, boxes)
        # for b, clases in enumerate(classes):
        #     lc = -1
        #     num = 0
        #     for i, c in enumerate(classes[b]):
        #         c = int(c)
        #         if lc == c:
        #             num+=1
        #         else:
        #             num = 0
        #             lc = c
        #         bx = box_dict[(b,c)][num]
        #         assert torch.abs(bx-boxes[b][i]).sum().cpu().item() < 1e-6

        cx = (squeezed_box[:, 0] + squeezed_box[:, 2]) / 2 / iw * ow
        cy = (squeezed_box[:, 1] + squeezed_box[:, 3]) / 2 / ih * oh
        cw = (squeezed_box[:, 2] - squeezed_box[:, 0]) / iw * ow
        ch = (squeezed_box[:, 3] - squeezed_box[:, 1]) / ih * oh
        # print("box", squeezed_box)
        # print("cx", cx)
        # print("cy", cy)
        # print("cw", cw)
        # print("ch", ch)
        # print("max h, w", torch.max(wh_pred))
        ceilx = cx.long()
        ceily = cy.long()
        deltx = cx - ceilx.float()
        delty = cy - ceily.float()
        del cx, cy
        anchor_ious = calc_iou_wh(
            cw, ch, anchor_shape[:, 0], anchor_shape[:, 1])
        # print("gbox anchor iou", anchor_ious)
        anchor_ids = torch.argmax(anchor_ious, dim=1)
        del anchor_ious

        cls_mask = torch.zeros((bs, oh, ow, na)).long().to(self.device)
        tcls = torch.zeros_like(cls_mask).long().to(self.device)
        i = 0
        for b in range(bs):
            for clas in classes[b]:
                cls_mask[b, ceily[i], ceilx[i], anchor_ids[i]] += 1
                tcls[b, ceily[i], ceilx[i], anchor_ids[i]] = clas
                i += 1
        cls_mask = (cls_mask == 1)
        # print("class mask sum", cls_mask.sum().item())
        # pred_cls:     b.num_cls * h * w * num_anchors
        pred_cls_reshape = cls_pred.reshape(
            bs, ncls, oh, ow, na).permute(0, 2, 3, 4, 1).contiguous()
        pred_cls_reshape = pred_cls_reshape[cls_mask].reshape(-1, ncls)
        tcls = tcls[cls_mask].reshape(-1)
        loss_cls = F.cross_entropy(
            pred_cls_reshape, tcls, reduction="sum") * self.cfg.lambda_class
        del cls_mask, tcls, pred_cls_reshape

        xy_pred = xy_pred[batch_class_mask]
        wh_pred = wh_pred[batch_class_mask]
        iou_pred = iou_pred[batch_class_mask]
        # iou_d = iou_pred.detach().reshape(nbc, oh, ow, na)

        with torch.no_grad():
            pcx = (xy_pred[:, :, :, :, 0] + torch.arange(ow).float().reshape(1,
                                                                        1, ow, 1).to(xy_pred.device)) * (iw/ow)
            pcy = (xy_pred[:, :, :, :, 1] + torch.arange(oh).float().reshape(1,
                                                                        oh, 1, 1).to(xy_pred.device)) * (ih/oh)
            pwh = wh_pred.reshape(
                nbc, oh, ow, na, 2) * anchor_shape * (ih/oh)
            pcx = pcx.reshape(nbc, oh*ow*na)
            pcy = pcy.reshape(nbc, oh*ow*na)
            pwh = pwh.reshape(nbc, oh*ow*na, 2)
            xmin = pcx - pwh[:, :, 0]/2
            xmax = pcx + pwh[:, :, 0]/2
            ymin = pcy - pwh[:, :, 1]/2
            ymax = pcy + pwh[:, :, 1]/2
            del pcx, pcy, pwh

            obj_mask = torch.zeros((nbc, oh, ow, na)).to(self.device)
            tiou = torch.zeros_like(obj_mask).to(self.device)
            coord_mask = torch.zeros((nbc, oh, ow, na)).to(self.device)
            tx = torch.zeros_like(coord_mask).to(self.device)
            ty = torch.zeros_like(coord_mask).to(self.device)
            tw = torch.ones_like(coord_mask).to(self.device)
            th = torch.ones_like(coord_mask).to(self.device)
            if self.seen < 12800:
                tx.fill_(0.5)
                ty.fill_(0.5)
                coord_mask.fill_(1)
                addition_wh = 8
            else:
                addition_wh = 1

            i = 0
            for (b, c), boxs in box_dict.items():
                pos = presum[b*ncls+c] - 1
                ious = calc_iou_xy(xmin[pos], ymin[pos], xmax[pos],
                                ymax[pos], boxs[:, 0], boxs[:, 1], boxs[:, 2], boxs[:, 3])
                # print(ious.shape)
                best_ious = torch.max(ious, dim=-1)[0].reshape(oh, ow, na)
                noobj_mask = best_ious < self.cfg.iou_thresh
                del best_ious
                obj_mask[pos, noobj_mask] = self.cfg.lambda_noobj
                # * iou_d[pos, noobj_mask]
                del noobj_mask
                ious = ious.reshape(oh, ow, na, boxs.shape[0])
                for bi, box in enumerate(boxs):
                    aid = anchor_ids[i]
                    obj_mask[pos, ceily[i], ceilx[i], aid] = self.cfg.lambda_obj
                    # * (1-iou_d[pos, ceily[i], ceilx[i], aid])
                    tiou[pos, ceily[i], ceilx[i],
                        aid] = ious[ceily[i], ceilx[i], aid, bi]
                    coord_mask[pos, ceily[i], ceilx[i],
                            aid] = 1
                    tx[pos, ceily[i], ceilx[i], aid] = deltx[i]
                    ty[pos, ceily[i], ceilx[i], aid] = delty[i]
                    tw[pos, ceily[i], ceilx[i], aid] = cw[i] / anchor_shape[aid][0]
                    th[pos, ceily[i], ceilx[i], aid] = ch[i] / anchor_shape[aid][1]
                    # print("wh", wh_pred[pos, ceily[i], ceilx[i], aid])
                    # print("w, h =", tw[pos, ceily[i], ceilx[i], aid].item(), th[pos, ceily[i], ceilx[i], aid].item(), wh_pred[pos, ceily[i], ceilx[i], aid])
                    # print("x, y =", tx[pos, ceily[i], ceilx[i], aid].item(), ty[pos, ceily[i], ceilx[i], aid].item(), xy_pred[pos, ceily[i], ceilx[i], aid])
                    # print("iou =", tiou[pos, ceilid, aid].item(), iou_pred[pos, ceily[i], ceilx[i], aid].item())
                    i += 1

            txy = torch.stack((tx, ty), dim=-1)
            twh = torch.stack((tw, th), dim=-1)
            del tx, ty, tw, th, ious
            coord_mask = coord_mask.unsqueeze(-1)
            obj_mask = obj_mask.sqrt()
        #################################################
        # max_mask = torch.max(wh_pred, dim=-1)[0] > 32
        # coord_mask[max_mask] = 0
        # print(nbox, bs)
        loss_coordxy = F.mse_loss(
            xy_pred * coord_mask, txy * coord_mask, reduction="sum") * self.cfg.lambda_coord / 2 # / nbox
        loss_coordwh = F.mse_loss(
            wh_pred.log() * coord_mask, twh.log() * coord_mask, reduction="sum") * self.cfg.lambda_coord / 2 / addition_wh # / nbox
        loss_obj = F.mse_loss(iou_pred * obj_mask, tiou *
                              obj_mask, reduction="sum") / 2 # / nbox
        loss_coord = loss_coordxy + loss_coordwh
        # fail_cnt = torch.sum(max_mask)
        # print("xy, wh loss:", loss_coordxy.item(), loss_coordwh.item())
        # print("obj, cls loss:", loss_obj.item(), loss_cls.item())
        # if fail_cnt > 0:
        #     print(f"{fail_cnt} wh are too large")

        return loss_cls, loss_coord, loss_obj, torch.sum(tiou > 0.5).cpu().item()/squeezed_box.size(0), nbox

    def pred_bbox(self, x, reweight_vector=None, iou_threshold=-1, nms_threshold=1):
        xy_pred, wh_pred, iou_pred, cls_pred = self.MetaLearner(
            x, reweight_vector)
        # pred: b.num_cls * h * w * num_anchors * 2
        # iou_pred, cls_pred: # b.num_cls * h * w * num_anchors
        wh_pred = torch.exp(wh_pred)
        bs = x.size(0)
        ncls = reweight_vector.size(1)
        ih, iw = x.shape[2:]
        oh, ow = iou_pred.shape[1:3]
        na = len(self.cfg.anchors)

        anchor_shape = torch.Tensor(self.cfg.anchors).to(self.device)

        cls_score = nn.functional.softmax(cls_pred.reshape(bs, ncls, oh, ow, na), dim=1)
        # pred_cls = torch.argmax(cls_score, dim=1, keepdim=True)
        # tmp = pred_cls.unsqueeze(-1).repeat(1, 1, 1, 1, 1, 2)
        # xy_pred = xy_pred.reshape(
        #     bs, ncls, oh, ow, na, 2).gather(1, tmp).squeeze(1)
        # wh_pred = wh_pred.reshape(
        #     bs, ncls, oh, ow, na, 2).gather(1, tmp).squeeze(1)
        # iou_pred = iou_pred.reshape(bs, ncls, oh, ow, na).gather(
        #     1, pred_cls).reshape(bs, oh*ow*na)
        # cls_score = torch.softmax(cls_score, dim=1).gather(
        #     1, pred_cls).reshape(bs, oh*ow*na)
        pred_cls = torch.arange(0,ncls,device=x.device).reshape(1,ncls,1,1,1).repeat(bs,1,oh,ow,na)
        pred_cls = pred_cls.reshape(bs, ncls, oh*ow*na)
        iou_pred = iou_pred.reshape(bs, ncls, oh*ow*na)
        cls_score = cls_score.reshape(bs, ncls, oh*ow*na)
        
        pcx = (xy_pred[:, :, :, :, 0] + torch.arange(ow).float().reshape(1,
                                                                         1, ow, 1).to(xy_pred.device)) * (iw/ow)
        pcy = (xy_pred[:, :, :, :, 1] + torch.arange(oh).float().reshape(1,
                                                                         oh, 1, 1).to(xy_pred.device)) * (ih/oh)
        pwh = wh_pred * anchor_shape * (ih/oh)
        pcx = pcx.reshape(bs, ncls, oh*ow*na)
        pcy = pcy.reshape(bs, ncls, oh*ow*na)
        pwh = pwh.reshape(bs, ncls, oh*ow*na, 2)

        xmin = pcx - pwh[:, :, :, 0]/2
        xmax = pcx + pwh[:, :, :, 0]/2
        ymin = pcy - pwh[:, :, :, 1]/2
        ymax = pcy + pwh[:, :, :, 1]/2

        ret = list()
        for i in range(bs):
            tc = list()
            for c in range(ncls):
                tc.append(nms(xmin[i, c], ymin[i, c], xmax[i, c], ymax[i, c], iou_pred[i, c],
                               cls_score[i, c], pred_cls[i, c], iou_threshold, nms_threshold))
            ntc = list()
            for j in range(6):
                kk = list(map(lambda x: x[j], tc))
                ntc.append(torch.cat(kk, dim=0))
            ret.append(ntc)
        return ret

    def optimize(self, x, boxes, classes, supp):
        if self.cfg.dynamic:
            loss_cls, loss_coord, loss_obj, correct_cnt, nbox = self.optimize_dynamic(
                x, boxes, classes, supp)
        else:
            loss_cls, loss_coord, loss_obj, correct_cnt = self.optimize_yolo(
                x, boxes, classes)
            nbox = 1

        self.seen += x.size(0)
        loss = (loss_cls + loss_coord + loss_obj)
        loss.backward()
        return loss_cls.cpu().item()/nbox, loss_coord.cpu().item()/nbox, loss_obj.cpu().item(), correct_cnt


def scatter_by_batch_class(classes, boxes):
    ret = dict()
    for b in range(len(classes)):
        for i, c in enumerate(classes[b]):
            c = int(c)
            if (b, c) not in ret:
                ret[(b, c)] = list()
            ret[(b, c)].append(boxes[b][i])
    for k, v in ret.items():
        ret[k] = torch.stack(v)
    return ret


def calc_iou_wh(a1, b1, a2, b2):
    ra1 = a1.reshape(-1, 1)
    ra2 = a2.reshape(1, -1)
    rb1 = b1.reshape(-1, 1)
    rb2 = b2.reshape(1, -1)
    sa = (ra1 < ra2).float()
    sb = (rb1 < rb2).float()
    ia = ra1*sa + ra2*(1-sa)
    ib = rb1*sb + rb2*(1-sb)
    iab = ia*ib
    return iab/(ra1*rb1+ra2*rb2-iab)


def exmin(a, b):
    a = a.reshape(-1, 1)
    b = b.reshape(1, -1)
    sig = (a < b).float()
    return sig*a + (1-sig)*b


def exmax(a, b):
    a = a.reshape(-1, 1)
    b = b.reshape(1, -1)
    sig = (a > b).float()
    return sig*a + (1-sig)*b


def calc_iou_xy(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    ixmin = exmax(xmin1, xmin2)
    ixmax = exmin(xmax1, xmax2)
    iymin = exmax(ymin1, ymin2)
    iymax = exmin(ymax1, ymax2)
    i = (ixmax-ixmin) * (iymax-iymin)
    i[ixmax < ixmin] = 0
    i[iymax < iymin] = 0
    s1 = (xmax1-xmin1) * (ymax1-ymin1)
    s2 = (xmax2-xmin2) * (ymax2-ymin2)
    u = s1.reshape(-1, 1) + s2.reshape(1, -1) - i
    return i/u


def nms(xmin, ymin, xmax, ymax, iou_pred, cls_score, pred_cls, iou_threshold, nms_threshold):
    score, idx = torch.sort(iou_pred*cls_score, descending=True)
    xmin = xmin[idx]
    ymin = ymin[idx]
    xmax = xmax[idx]
    ymax = ymax[idx]
    iou_pred = iou_pred[idx]
    pred_cls = pred_cls[idx]

    filt = (iou_pred >= iou_threshold)
    bb_ious = calc_iou_xy(xmin, ymin, xmax, ymax, xmin, ymin, xmax, ymax)
    for i in range(len(xmin)):
        if not filt[i].item():
            continue
        filt[i+1:] &= (bb_ious[i, i+1:] <
                       nms_threshold) | (pred_cls[i+1:] != pred_cls[i])
    xmin = xmin[filt]
    ymin = ymin[filt]
    xmax = xmax[filt]
    ymax = ymax[filt]
    iou_pred = iou_pred[filt]
    pred_cls = pred_cls[filt]
    score = score[filt]

    return xmin, ymin, xmax, ymax, pred_cls, score


def optimize_yolo(self, x, boxes, classes):
    # x: b x 3 x h x w
    # boxes: b x [? x 4]    sorted by class
    # classes: b x [?]      sorted by class
    xy_pred, wh_pred, iou_pred, cls_pred = self.forward(x)
    wh_pred = wh_pred.exp()
    # pred: b * h * w * num_anchors * 2
    # iou_pred: b * h * w * num_anchors
    # cls_pred: b * h * w * num_anchors * num_cls
    bs = x.size(0)
    ncls = 15
    ih, iw = x.shape[2:]
    oh, ow = iou_pred.shape[1:3]
    na = len(self.cfg.anchors)
    nbox = sum(map(len, classes))
    anchor_shape = torch.Tensor(self.cfg.anchors).to(self.device)
    cls_mask = torch.zeros((bs, oh, ow, na)).long().to(self.device)
    tcls = torch.zeros_like(cls_mask).long().to(self.device)
    iou_d = iou_pred.detach().reshape(-1, oh, ow, na)

    pcxy = xy_pred.detach()
    pcx = (pcxy[:, :, :, :, 0] + torch.arange(ow).float().reshape(1,
                                                                  1, ow, 1).to(pcxy.device)) * (iw/ow)
    pcy = (pcxy[:, :, :, :, 1] + torch.arange(oh).float().reshape(1,
                                                                  oh, 1, 1).to(pcxy.device)) * (ih/oh)
    pwh = wh_pred.detach().reshape(
        bs, oh*ow, na, 2) * anchor_shape * (ih/oh)

    pcx = pcx.reshape(bs, oh*ow*na)
    pcy = pcy.reshape(bs, oh*ow*na)
    pwh = pwh.reshape(bs, oh*ow*na, 2)
    xmin = pcx - pwh[:, :, 0]/2
    xmax = pcx + pwh[:, :, 0]/2
    ymin = pcy - pwh[:, :, 1]/2
    ymax = pcy + pwh[:, :, 1]/2

    obj_mask = torch.zeros((bs, oh, ow, na)).to(self.device)
    tiou = torch.zeros_like(obj_mask).to(self.device)
    coord_mask = torch.zeros((bs, oh, ow, na)).to(self.device)
    tx = torch.zeros_like(coord_mask).to(self.device)
    ty = torch.zeros_like(coord_mask).to(self.device)
    tw = torch.zeros_like(coord_mask).to(self.device)
    th = torch.zeros_like(coord_mask).to(self.device)
    tx.fill_(0.5)
    ty.fill_(0.5)
    tw.fill_(1)
    th.fill_(1)
    coord_mask.fill_(0.01)

    for b in range(bs):
        if self.cfg.debug:
            plt.cla()
            fig = plt.gcf()
            ax = plt.gca()
            ax.axis('off')
            input_img = x[b].permute((1, 2, 0)).cpu().numpy()
            ax.imshow(input_img)
        cx = (boxes[b][:, 0] + boxes[b][:, 2]) / 2 / iw * ow
        cy = (boxes[b][:, 1] + boxes[b][:, 3]) / 2 / ih * oh
        cw = (boxes[b][:, 2] - boxes[b][:, 0]) / iw * ow
        ch = (boxes[b][:, 3] - boxes[b][:, 1]) / ih * oh
        # print("box", squeezed_box)
        # print("cx", cx)
        # print("cy", cy)
        # print("cw", cw)
        # print("ch", ch)
        ceilx = torch.floor(cx).long()
        ceily = torch.floor(cy).long()
        deltx = cx - ceilx.float()
        delty = cy - ceily.float()
        anchor_ious = calc_iou_wh(
            cw, ch, anchor_shape[:, 0], anchor_shape[:, 1])
        # print("gbox anchor iou", anchor_ious)
        anchor_ids = torch.argmax(anchor_ious, dim=1)

        # print("class mask sum", cls_mask.sum().item())
        # pred_cls:     b.num_cls * h * w * num_anchors

        boxs = boxes[b]
        ious = calc_iou_xy(xmin[b], ymin[b], xmax[b],
                           ymax[b], boxs[:, 0], boxs[:, 1], boxs[:, 2], boxs[:, 3])
        best_ious = torch.max(ious, dim=1)[0].reshape(oh, ow, na)
        ious = ious.reshape(oh, ow, na, boxs.shape[0])
        noobj_mask = best_ious < self.cfg.iou_thresh
        obj_mask[b, noobj_mask] = self.cfg.lambda_noobj * \
            (0-iou_d[b, noobj_mask])  # ??????????????

        for i in range(len(boxs)):
            # assert torch.abs(box-squeezed_box[i]).sum().item() < 1e-6
            # print(b, i, ceily.shape, ceilx.shape, anchor_ids.shape)
            # print(b, ceily[i], ceilx[i], anchor_ids[i])
            cls_mask[b, ceily[i], ceilx[i], anchor_ids[i]] += 1
            tcls[b, ceily[i], ceilx[i], anchor_ids[i]] = classes[b][i]

            aid = anchor_ids[i]
            obj_mask[b, ceily[i], ceilx[i], aid] = (
                1-iou_d[b, ceily[i], ceilx[i], aid])*self.cfg.lambda_obj  # ??????????????
            tiou[b, ceily[i], ceilx[i], aid] = ious[ceily[i], ceilx[i], aid, i]
            # print(f"{pos},{ceilid},{aid} iou = {ious[ceilid*na+aid, bi]}")
            coord_mask[b, ceily[i], ceilx[i], aid] = self.cfg.lambda_coord
            tx[b, ceily[i], ceilx[i], aid] = deltx[i]
            ty[b, ceily[i], ceilx[i], aid] = delty[i]
            tw[b, ceily[i], ceilx[i], aid] = cw[i] / anchor_shape[aid][0]
            th[b, ceily[i], ceilx[i], aid] = ch[i] / anchor_shape[aid][1]
            if self.cfg.debug:
                sx = float(deltx[i] + ceilx[i]) * iw/ow
                sy = float(delty[i] + ceily[i]) * ih/oh
                sw = float(cw[i]) * iw/ow
                sh = float(ch[i]) * ih/oh
                rect = mpatches.Rectangle(
                    (sx-sw/2, sy-sh/2), sw, sh, color="red", fill=False)
                ax.add_patch(rect)
                ax.annotate(str(int(classes[b][i])), (sx-sw/2, sy-sh/2))
            # print("wh", wh_pred[pos, ceily[i], ceilx[i], aid])
            # print("w, h =", tw[pos, ceilid, aid].item(), th[pos, ceilid, aid].item(), wh_pred[pos, ceily[i], ceilx[i], aid])
            # print("x, y =", tx[pos, ceilid, aid].item(), ty[pos, ceilid, aid].item(), xy_pred[pos, ceily[i], ceilx[i], aid])
            # print("iou =", tiou[pos, ceilid, aid].item(), iou_pred[pos, ceily[i], ceilx[i], aid].item())

        if self.cfg.debug:
            plt.show()

    cls_mask = (cls_mask == 1)
    txy = torch.stack((tx, ty), dim=-1)
    twh = torch.stack((tw, th), dim=-1)
    coord_mask = coord_mask.unsqueeze(-1)
    # obj_mask = obj_mask.sqrt()
    # print(xy_pred.reshape(nbc,oh*ow,na,2) - coord_mask, txy * coord_mask)
    pred_cls_reshape = cls_pred.reshape(
        bs, oh, ow, na, ncls).contiguous()
    pred_cls_reshape = pred_cls_reshape[cls_mask].reshape(-1, ncls)
    tcls = tcls[cls_mask].reshape(-1)
    loss_cls = F.cross_entropy(
        pred_cls_reshape, tcls) * self.cfg.lambda_class
    loss_coordxy = F.mse_loss(
        xy_pred * coord_mask, txy * coord_mask, reduction="sum") / nbox
    loss_coordwh = F.mse_loss(
        wh_pred.log() * coord_mask, twh.log() * coord_mask, reduction="sum") / nbox
    loss_obj = F.mse_loss(iou_pred * obj_mask, tiou *
                          obj_mask, reduction="sum") / nbox
    loss_coord = loss_coordxy + loss_coordwh
    # print(f"xy:{loss_coordxy.cpu().item()}, wh:{loss_coordwh.cpu().item()}")

    assert not txy.requires_grad
    assert not twh.requires_grad
    assert not coord_mask.requires_grad
    assert not tiou.requires_grad
    assert not obj_mask.requires_grad

    return loss_cls, loss_coord, loss_obj, torch.sum(tiou > 0.5).cpu().item()/nbox
