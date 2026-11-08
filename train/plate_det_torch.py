"""Plate detector training from random initialisation -- no Ultralytics code, no Ultralytics weights.

Why this file exists: the shipped detectors (models/plate_det_v8n_*.onnx in yolo_lpr_cpp) were
fine-tuned from Ultralytics' yolov8n.pt, so they inherit AGPL-3.0. This trains the same
architecture from scratch with code written here (BSD-3-Clause, like the C++ engines it mirrors):

  * model  -- YOLOv8n written out by hand. Parameter names follow the Ultralytics state_dict layout
             (`model.0.conv.weight`, `model.22.cv3.2.2.bias`, ...) only because that is the layout
             yolo_lpr_cpp's C++ graph writer expects: `jlpr init-det --from-pt best.pt` turns the
             state_dict this script saves into the ONNX the analyzer runs. No weights are loaded.
  * loss   -- task-aligned assignment + CIoU + DFL + BCE, the published YOLOv8 recipe, written
             from the paper / the BSD C++ ports in yolov8_cpp (pure/tal.hpp, pure/v8pure.hpp) and
             yolo_lpr_cpp (pure/train_det.hpp).
  * data   -- YOLO-format dirs (images/*.png, labels/*.txt `0 xc yc w h`), e.g. `jlpr gen-det`.

  python train/plate_det_torch.py --data data/det_a --data data/det_b --val data/det_val \
      --epochs 150 --batch 32 --imgsz 640 --out runs/scratch
  -> runs/scratch/best.pt, last.pt (EMA weights, float32 state_dict) + log.csv
"""
import argparse
import csv
import glob
import math
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

cv2.setNumThreads(0)

# ----------------------------------------------------------------------------- model


class Conv(nn.Module):
    """conv (no bias) + BN + SiLU."""

    def __init__(self, c1, c2, k=1, s=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c, shortcut):
        super().__init__()
        self.cv1 = Conv(c, c, 3)
        self.cv2 = Conv(c, c, 3)
        self.add = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    def __init__(self, c1, c2, n, shortcut):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, shortcut) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        c = c1 // 2
        self.cv1 = Conv(c1, c, 1)
        self.cv2 = Conv(c * 4, c2, 1)
        self.m = nn.MaxPool2d(5, 1, 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class DFL(nn.Module):
    def __init__(self, reg=16):
        super().__init__()
        self.conv = nn.Conv2d(reg, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = torch.arange(reg, dtype=torch.float).view(1, reg, 1, 1)


class Detect(nn.Module):
    def __init__(self, nc, ch, reg=16):
        super().__init__()
        self.nc, self.reg = nc, reg
        c2, c3 = max(16, ch[0] // 4, reg * 4), max(ch[0], min(nc, 100))
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, nc, 1)) for x in ch)
        self.dfl = DFL(reg)

    def forward(self, xs):
        return [(self.cv2[i](x), self.cv3[i](x)) for i, x in enumerate(xs)]

    def bias_init(self, strides, imgsz=640):
        for a, b, s in zip(self.cv2, self.cv3, strides):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (imgsz / s) ** 2)


class YOLOv8(nn.Module):
    """yolov8 n/s by width & depth multiples; `model.<i>` indices as in the reference yaml."""

    STRIDES = (8, 16, 32)

    def __init__(self, nc=1, width=0.25, depth=0.33, max_ch=1024):
        super().__init__()
        ch = lambda c: int(math.ceil(min(c, max_ch) * width / 8) * 8)  # noqa: E731
        rep = lambda n: max(round(n * depth), 1)  # noqa: E731
        c1, c2, c3, c4, c5 = ch(64), ch(128), ch(256), ch(512), ch(1024)
        L = [
            Conv(3, c1, 3, 2), Conv(c1, c2, 3, 2), C2f(c2, c2, rep(3), True),        # 0-2
            Conv(c2, c3, 3, 2), C2f(c3, c3, rep(6), True),                           # 3-4  (P3)
            Conv(c3, c4, 3, 2), C2f(c4, c4, rep(6), True),                           # 5-6  (P4)
            Conv(c4, c5, 3, 2), C2f(c5, c5, rep(3), True), SPPF(c5, c5),             # 7-9  (P5)
            nn.Upsample(None, 2, "nearest"), nn.Identity(), C2f(c5 + c4, c4, rep(3), False),   # 10-12
            nn.Upsample(None, 2, "nearest"), nn.Identity(), C2f(c4 + c3, c3, rep(3), False),   # 13-15 out P3
            Conv(c3, c3, 3, 2), nn.Identity(), C2f(c3 + c4, c4, rep(3), False),      # 16-18 out P4
            Conv(c4, c4, 3, 2), nn.Identity(), C2f(c4 + c5, c5, rep(3), False),      # 19-21 out P5
            Detect(nc, (c3, c4, c5)),                                                # 22
        ]
        self.model = nn.ModuleList(L)
        self.nc = nc
        self.model[22].bias_init(self.STRIDES)

    def forward(self, x):
        m = self.model
        x = m[1](m[0](x))
        x = m[2](x)
        p3 = m[4](m[3](x))
        p4 = m[6](m[5](p3))
        p5 = m[9](m[8](m[7](p4)))
        h = m[12](torch.cat([m[10](p5), p4], 1))
        o3 = m[15](torch.cat([m[13](h), p3], 1))
        o4 = m[18](torch.cat([m[16](o3), h], 1))
        o5 = m[21](torch.cat([m[19](o4), p5], 1))
        return m[22]([o3, o4, o5])


# ----------------------------------------------------------------------------- loss


def make_anchors(feats, strides, offset=0.5):
    pts, sts = [], []
    for (box, _), s in zip(feats, strides):
        h, w = box.shape[2:]
        sy, sx = torch.meshgrid(torch.arange(h, device=box.device, dtype=torch.float32) + offset,
                                torch.arange(w, device=box.device, dtype=torch.float32) + offset, indexing="ij")
        pts.append(torch.stack((sx, sy), -1).view(-1, 2))
        sts.append(torch.full((h * w, 1), float(s), device=box.device))
    return torch.cat(pts), torch.cat(sts)


def bbox_iou_ciou(b1, b2, eps=1e-7):
    """CIoU of xyxy boxes, elementwise over the last dim."""
    x1, y1, x2, y2 = b1.unbind(-1)
    X1, Y1, X2, Y2 = b2.unbind(-1)
    w1, h1, w2, h2 = x2 - x1, y2 - y1 + eps, X2 - X1, Y2 - Y1 + eps
    inter = (torch.min(x2, X2) - torch.max(x1, X1)).clamp(0) * (torch.min(y2, Y2) - torch.max(y1, Y1)).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    cw = torch.max(x2, X2) - torch.min(x1, X1)
    ch = torch.max(y2, Y2) - torch.min(y1, Y1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((X1 + X2 - x1 - x2) ** 2 + (Y1 + Y2 - y1 - y2) ** 2) / 4
    v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


def box_iou_pairwise(gt, pd, eps=1e-7):
    """gt (B,M,4) x pd (B,A,4) -> (B,M,A) plain IoU."""
    g = gt.unsqueeze(2)
    p = pd.unsqueeze(1)
    inter = (torch.min(g[..., 2], p[..., 2]) - torch.max(g[..., 0], p[..., 0])).clamp(0) * \
            (torch.min(g[..., 3], p[..., 3]) - torch.max(g[..., 1], p[..., 1])).clamp(0)
    ag = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
    ap = (p[..., 2] - p[..., 0]) * (p[..., 3] - p[..., 1])
    return inter / (ag + ap - inter + eps)


@torch.no_grad()
def tal_assign(pd_scores, pd_boxes, anc, gt_labels, gt_boxes, mask_gt, nc, topk=10, alpha=0.5, beta=6.0, eps=1e-9):
    """Task-aligned assignment. pd_* in image pixels; returns target boxes/scores and the fg mask."""
    B, A, _ = pd_scores.shape
    M = gt_boxes.shape[1]
    if M == 0:
        return (torch.zeros_like(pd_boxes), torch.zeros_like(pd_scores), torch.zeros(B, A, dtype=torch.bool, device=pd_scores.device))
    # anchors whose centre lies inside the gt box
    lt = anc.view(1, 1, A, 2) - gt_boxes[..., :2].unsqueeze(2)
    rb = gt_boxes[..., 2:].unsqueeze(2) - anc.view(1, 1, A, 2)
    in_gt = torch.cat([lt, rb], -1).amin(-1).gt_(eps)                        # (B,M,A)
    valid = in_gt & mask_gt.bool()
    ious = box_iou_pairwise(gt_boxes, pd_boxes).clamp_(0)                   # (B,M,A)
    cls_idx = gt_labels.long().clamp(0, nc - 1)                             # (B,M)
    sc = pd_scores.gather(2, cls_idx.unsqueeze(1).expand(B, A, M)).permute(0, 2, 1)  # (B,M,A)
    ious = ious * valid
    sc = sc * valid
    align = sc.pow(alpha) * ious.pow(beta)
    k = min(topk, A)
    top = align.topk(k, dim=-1).indices                                     # (B,M,k)
    is_top = torch.zeros_like(align, dtype=torch.bool).scatter_(-1, top, True)
    pos = is_top & valid                                                    # (B,M,A)
    # an anchor claimed by several gts goes to the one it overlaps most
    multi = pos.sum(1, keepdim=True) > 1
    if multi.any():
        best = ious.argmax(1, keepdim=True)
        only_best = torch.zeros_like(pos).scatter_(1, best, True)
        pos = torch.where(multi.expand_as(pos), only_best & pos.any(1, keepdim=True).expand_as(pos), pos)
    fg = pos.any(1)                                                         # (B,A)
    gt_idx = pos.float().argmax(1)                                          # (B,A)
    bidx = torch.arange(B, device=gt_boxes.device).unsqueeze(1)
    t_boxes = gt_boxes[bidx, gt_idx]                                        # (B,A,4)
    t_lab = cls_idx[bidx, gt_idx]                                           # (B,A)
    t_scores = F.one_hot(t_lab, nc).float() * fg.unsqueeze(-1)
    # normalise: each gt's best anchor gets that gt's best IoU as its target score
    align = align * pos
    pos_align = align.amax(-1, keepdim=True)
    pos_iou = (ious * pos).amax(-1, keepdim=True)
    norm = (align * pos_iou / (pos_align + eps)).amax(1).unsqueeze(-1)       # (B,A,1)
    return t_boxes, t_scores * norm, fg


class V8Loss:
    def __init__(self, nc, reg=16, strides=(8, 16, 32), box=7.5, cls=0.5, dfl=1.5):
        self.nc, self.reg, self.strides = nc, reg, strides
        self.gain = (box, cls, dfl)
        self.proj = None

    def __call__(self, feats, targets, imgsz):
        """targets: (N,6) = batch_idx, cls, xc, yc, w, h normalised."""
        dev = feats[0][0].device
        B = feats[0][0].shape[0]
        pd_dist = torch.cat([b.flatten(2) for b, _ in feats], 2).permute(0, 2, 1)   # (B,A,4*reg)
        pd_cls = torch.cat([c.flatten(2) for _, c in feats], 2).permute(0, 2, 1)    # (B,A,nc)
        anc, st = make_anchors(feats, self.strides)
        if self.proj is None or self.proj.device != dev:
            self.proj = torch.arange(self.reg, dtype=torch.float32, device=dev)
        A = anc.shape[0]
        # gt -> (B, M, 4) xyxy pixels
        counts = torch.bincount(targets[:, 0].long(), minlength=B) if len(targets) else torch.zeros(B, dtype=torch.long)
        M = int(counts.max()) if len(targets) else 0
        gt_boxes = torch.zeros(B, M, 4, device=dev)
        gt_lab = torch.zeros(B, M, device=dev)
        mask = torch.zeros(B, M, 1, device=dev)
        if M:
            slot = torch.zeros(B, dtype=torch.long)
            for t in targets.cpu():
                b = int(t[0]); j = int(slot[b]); slot[b] += 1
                xc, yc, w, h = (t[2:] * imgsz).tolist()
                gt_boxes[b, j] = torch.tensor([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
                gt_lab[b, j] = t[1]
                mask[b, j] = 1
        # decode predictions (grid units)
        d = pd_dist.float().view(B, A, 4, self.reg).softmax(3) @ self.proj
        lt, rb = d[..., :2], d[..., 2:]
        pd_boxes = torch.cat([anc - lt, anc + rb], -1)
        t_boxes, t_scores, fg = tal_assign(pd_cls.detach().float().sigmoid(), (pd_boxes.detach() * st),
                                           anc * st, gt_lab, gt_boxes, mask, self.nc)
        tss = max(t_scores.sum(), 1.0)
        l_cls = F.binary_cross_entropy_with_logits(pd_cls.float(), t_scores, reduction="sum") / tss
        l_box = torch.zeros((), device=dev)
        l_dfl = torch.zeros((), device=dev)
        if fg.any():
            w = t_scores.sum(-1)[fg].unsqueeze(-1)
            tb = t_boxes / st                                                 # back to grid units
            iou = bbox_iou_ciou(pd_boxes[fg], tb[fg])
            l_box = ((1.0 - iou).unsqueeze(-1) * w).sum() / tss
            a = anc.unsqueeze(0).expand(B, A, 2)[fg]
            tgt = torch.cat([a - tb[fg][:, :2], tb[fg][:, 2:] - a], -1).clamp_(0, self.reg - 1 - 0.01)
            tl = tgt.long()
            tr = tl + 1
            wl = tr - tgt
            wr = 1 - wl
            pdf = pd_dist[fg].float().view(-1, self.reg)
            l = F.cross_entropy(pdf, tl.view(-1), reduction="none").view(tl.shape) * wl + \
                F.cross_entropy(pdf, tr.view(-1), reduction="none").view(tl.shape) * wr
            l_dfl = (l.mean(-1, keepdim=True) * w).sum() / tss
        g = self.gain
        loss = l_box * g[0] + l_cls * g[1] + l_dfl * g[2]
        return loss * B, torch.stack([l_box, l_cls, l_dfl]).detach()


# ----------------------------------------------------------------------------- data


def list_items(roots):
    items = []
    for r in roots:
        for p in sorted(glob.glob(os.path.join(r, "images", "*"))):
            if not p.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            lab = os.path.join(r, "labels", os.path.splitext(os.path.basename(p))[0] + ".txt")
            items.append((p, lab))
    return items


def read_labels(path):
    if not os.path.exists(path):
        return np.zeros((0, 5), np.float32)
    rows = [l.split() for l in open(path) if l.strip()]
    return np.array(rows, np.float32).reshape(-1, 5)


def hsv_aug(img, hg=0.015, sg=0.7, vg=0.4):
    r = np.random.uniform(-1, 1, 3) * [hg, sg, vg] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    x = np.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * r[2], 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v))), cv2.COLOR_HSV2BGR)


class DetData(torch.utils.data.Dataset):
    def __init__(self, items, imgsz, train, mosaic=1.0, scale=0.5, degrees=5.0, translate=0.1, blur=0.2):
        self.items, self.S, self.train = items, imgsz, train
        self.mosaic, self.scale, self.degrees, self.translate, self.blur = mosaic, scale, degrees, translate, blur

    def __len__(self):
        return len(self.items)

    def load(self, i):
        p, lab = self.items[i]
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        r = self.S / max(h, w)
        if r != 1:
            img = cv2.resize(img, (max(1, round(w * r)), max(1, round(h * r))), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        lb = read_labels(lab)
        h, w = img.shape[:2]
        boxes = np.zeros((len(lb), 4), np.float32)  # xyxy pixels
        if len(lb):
            boxes[:, 0] = (lb[:, 1] - lb[:, 3] / 2) * w
            boxes[:, 1] = (lb[:, 2] - lb[:, 4] / 2) * h
            boxes[:, 2] = (lb[:, 1] + lb[:, 3] / 2) * w
            boxes[:, 3] = (lb[:, 2] + lb[:, 4] / 2) * h
        return img, boxes, lb[:, 0] if len(lb) else np.zeros(0, np.float32)

    def letterbox(self, img, boxes):
        S = self.S
        h, w = img.shape[:2]
        canvas = np.full((S, S, 3), 114, np.uint8)
        top, left = (S - h) // 2, (S - w) // 2
        canvas[top:top + h, left:left + w] = img
        b = boxes.copy()
        b[:, [0, 2]] += left
        b[:, [1, 3]] += top
        return canvas, b

    def mosaic4(self, i):
        S = self.S
        yc, xc = (int(random.uniform(S * 0.5, S * 1.5)) for _ in range(2))
        canvas = np.full((S * 2, S * 2, 3), 114, np.uint8)
        all_b, all_c = [], []
        ids = [i] + random.choices(range(len(self.items)), k=3)
        for k, j in enumerate(ids):
            img, b, c = self.load(j)
            h, w = img.shape[:2]
            if k == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif k == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, S * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif k == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(S * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, S * 2), min(S * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            b = b.copy()
            b[:, [0, 2]] += x1a - x1b
            b[:, [1, 3]] += y1a - y1b
            all_b.append(b)
            all_c.append(c)
        return canvas, np.concatenate(all_b), np.concatenate(all_c)

    def affine(self, img, boxes, border):
        S = self.S
        h, w = img.shape[:2]
        C = np.eye(3)
        C[0, 2], C[1, 2] = -w / 2, -h / 2
        R = np.eye(3)
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D((0, 0), a, s)
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * S
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * S
        Mx = T @ R @ C
        out = cv2.warpAffine(img, Mx[:2], (S, S), borderValue=(114, 114, 114))
        if len(boxes):
            n = len(boxes)
            pts = np.ones((n * 4, 3))
            pts[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
            pts = (pts @ Mx.T)[:, :2].reshape(n, 8)
            xs, ys = pts[:, [0, 2, 4, 6]], pts[:, [1, 3, 5, 7]]
            nb = np.stack([xs.min(1), ys.min(1), xs.max(1), ys.max(1)], 1)
            # rotation inflates a box's corners; shrink toward the original area like the reference
            nb[:, [0, 2]] = nb[:, [0, 2]].clip(0, S)
            nb[:, [1, 3]] = nb[:, [1, 3]].clip(0, S)
            boxes = nb.astype(np.float32)
        return out, boxes

    def __getitem__(self, i):
        if self.train and random.random() < self.mosaic:
            img, boxes, cls = self.mosaic4(i)
            img, nb = self.affine(img, boxes, border=self.S // 2)
        else:
            img, boxes, cls = self.load(i)
            img, boxes = self.letterbox(img, boxes)
            nb = boxes
            if self.train:
                img, nb = self.affine(img, boxes, border=0)
        if self.train:
            # drop boxes that lost most of themselves at the edge or became slivers
            if len(nb):
                w, h = nb[:, 2] - nb[:, 0], nb[:, 3] - nb[:, 1]
                keep = (w > 3) & (h > 2) & (w / np.maximum(h, 1e-3) < 20)
                nb, cls = nb[keep], cls[keep]
            img = hsv_aug(img)
            if random.random() < self.blur:  # motion / focus blur: video frames are rarely sharp
                k = random.choice([3, 5])
                img = cv2.blur(img, (k, 1) if random.random() < 0.5 else (k, k))
        S = self.S
        lab = np.zeros((len(nb), 6), np.float32)
        if len(nb):
            lab[:, 1] = cls
            lab[:, 2] = (nb[:, 0] + nb[:, 2]) / 2 / S
            lab[:, 3] = (nb[:, 1] + nb[:, 3]) / 2 / S
            lab[:, 4] = (nb[:, 2] - nb[:, 0]) / S
            lab[:, 5] = (nb[:, 3] - nb[:, 1]) / S
        x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)))  # BGR -> RGB, CHW uint8
        return x, torch.from_numpy(lab)


def collate(batch):
    xs, labs = zip(*batch)
    for i, l in enumerate(labs):
        l[:, 0] = i
    return torch.stack(xs), torch.cat(labs)


# ----------------------------------------------------------------------------- eval


@torch.no_grad()
def decode(feats, strides, reg, conf=0.001, iou=0.6, max_det=100):
    import torchvision
    anc, st = make_anchors(feats, strides)
    B = feats[0][0].shape[0]
    dist = torch.cat([b.flatten(2) for b, _ in feats], 2).permute(0, 2, 1).float()
    cls = torch.cat([c.flatten(2) for _, c in feats], 2).permute(0, 2, 1).float().sigmoid()
    A = anc.shape[0]
    d = dist.view(B, A, 4, reg).softmax(3) @ torch.arange(reg, device=dist.device, dtype=torch.float32)
    boxes = torch.cat([anc - d[..., :2], anc + d[..., 2:]], -1) * st
    out = []
    for b in range(B):
        s, _ = cls[b].max(-1)
        k = s > conf
        bb, ss = boxes[b][k], s[k]
        keep = torchvision.ops.nms(bb, ss, iou)[:max_det]
        out.append((bb[keep], ss[keep]))
    return out


def ap50(preds, gts):
    """preds/gts per image: (boxes xyxy, scores) / boxes. Returns (AP@0.5, recall@conf0.25)."""
    import torchvision
    scores, tp, ngt, rec_hit = [], [], 0, 0
    for (pb, ps), gb in zip(preds, gts):
        ngt += len(gb)
        if len(pb) == 0:
            continue
        matched = torch.zeros(len(gb), dtype=torch.bool)
        order = ps.argsort(descending=True)
        ious = torchvision.ops.box_iou(pb[order], gb) if len(gb) else torch.zeros(len(pb), 0)
        for r, i in enumerate(order):
            ok = False
            if len(gb):
                v, j = ious[r].max(0)
                if v >= 0.5 and not matched[j]:
                    matched[j] = True
                    ok = True
                    if ps[i] >= 0.25:
                        rec_hit += 1
            scores.append(float(ps[i]))
            tp.append(ok)
    if ngt == 0:
        return 0.0, 0.0
    o = np.argsort(-np.array(scores))
    tpc = np.cumsum(np.array(tp)[o])
    fpc = np.cumsum(~np.array(tp)[o])
    rec = tpc / ngt
    prec = tpc / np.maximum(tpc + fpc, 1)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[1], prec, [0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x)), rec_hit / ngt


@torch.no_grad()
def evaluate(model, loader, device, reg=16):
    model.eval()
    preds, gts = [], []
    for x, lab in loader:
        x = x.to(device, non_blocking=True).float() / 255
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            feats = model(x)
        for b, (pb, ps) in enumerate(decode(feats, YOLOv8.STRIDES, reg)):
            preds.append((pb.cpu(), ps.cpu()))
            l = lab[lab[:, 0] == b]
            S = x.shape[2]
            g = torch.stack([(l[:, 2] - l[:, 4] / 2) * S, (l[:, 3] - l[:, 5] / 2) * S,
                             (l[:, 2] + l[:, 4] / 2) * S, (l[:, 3] + l[:, 5] / 2) * S], 1) if len(l) else torch.zeros(0, 4)
            gts.append(g)
    model.train()
    return ap50(preds, gts)


# ----------------------------------------------------------------------------- train


class EMA:
    def __init__(self, model, decay=0.9999, tau=2000):
        import copy
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.n = decay, tau, 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = self.decay * (1 - math.exp(-self.n / self.tau))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)


def save_state(model, path):
    sd = {k: v.detach().float().cpu().contiguous() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
    torch.save(sd, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--close-mosaic", type=int, default=10)
    ap.add_argument("--out", default="runs/scratch")
    ap.add_argument("--resume", default="")
    ap.add_argument("--max-hours", type=float, default=0, help="stop (and save) after this much wall time")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_items, va_items = list_items(args.data), list_items([args.val])
    print(f"train {len(tr_items)} frames, val {len(va_items)} frames, device {dev}", flush=True)
    tr_ds = DetData(tr_items, args.imgsz, True)
    va_ds = DetData(va_items, args.imgsz, False)
    tr = torch.utils.data.DataLoader(tr_ds, args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate,
                                     pin_memory=True, drop_last=True)  # not persistent: close-mosaic must reach the workers
    va = torch.utils.data.DataLoader(va_ds, args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate)

    model = YOLOv8(nc=1).to(dev)
    print("parameters:", sum(p.numel() for p in model.parameters()), flush=True)
    # SGD, no weight decay on BN / biases (standard practice)
    g_bn, g_w, g_b = [], [], []
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            if pn == "bias":
                g_b.append(p)
            elif isinstance(m, nn.BatchNorm2d):
                g_bn.append(p)
            else:
                g_w.append(p)
    opt = torch.optim.SGD([{"params": g_bn, "weight_decay": 0.0}, {"params": g_w, "weight_decay": 5e-4 * args.batch / 64},
                           {"params": g_b, "weight_decay": 0.0}], lr=args.lr, momentum=0.937, nesterov=True)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    ema = EMA(model)
    crit = V8Loss(nc=1)
    nb = len(tr)
    warm = max(3 * nb, 300)
    lf = lambda e: (1 - e / args.epochs) * (1 - 0.01) + 0.01  # noqa: E731  linear to lr*0.01
    start, best = 0, -1.0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ck["model"])
        ema.ema.load_state_dict(ck["ema"])
        ema.n = ck["ema_n"]
        opt.load_state_dict(ck["opt"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed at epoch {start}, best AP50 {best:.4f}", flush=True)
    log = open(os.path.join(args.out, "log.csv"), "a", newline="")
    w = csv.writer(log)
    t_start = time.time()
    for ep in range(start, args.epochs):
        if ep == args.epochs - args.close_mosaic:
            tr_ds.mosaic = 0.0
            print("closing mosaic", flush=True)
        model.train()
        t0 = time.time()
        msum = torch.zeros(3)
        for i, (x, lab) in enumerate(tr):
            it = i + nb * ep
            x = x.to(dev, non_blocking=True).float() / 255
            if it <= warm:
                for j, g in enumerate(opt.param_groups):
                    g["lr"] = np.interp(it, [0, warm], [0.1 if j == 2 else 0.0, args.lr * lf(ep)])
                    g["momentum"] = np.interp(it, [0, warm], [0.8, 0.937])
            else:
                for g in opt.param_groups:
                    g["lr"] = args.lr * lf(ep + i / nb)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                feats = model(x)
            loss, parts = crit(feats, lab.to(dev), args.imgsz)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            ema.update(model)
            msum += parts.cpu()
            if i % 50 == 0:
                print(f"ep {ep} it {i}/{nb} box {parts[0]:.3f} cls {parts[1]:.3f} dfl {parts[2]:.3f} "
                      f"{(time.time() - t0) / (i + 1):.2f}s/it", flush=True)
        m = msum / nb
        apv, rec = evaluate(ema.ema, va, dev)
        dt = time.time() - t0
        print(f"== epoch {ep} loss box {m[0]:.3f} cls {m[1]:.3f} dfl {m[2]:.3f} | val AP50 {apv:.4f} "
              f"recall@0.25 {rec:.4f} | {dt:.0f}s", flush=True)
        w.writerow([ep, float(m[0]), float(m[1]), float(m[2]), apv, rec, round(dt)])
        log.flush()
        save_state(ema.ema, os.path.join(args.out, "last.pt"))
        if apv > best:
            best = apv
            save_state(ema.ema, os.path.join(args.out, "best.pt"))
        torch.save({"model": model.state_dict(), "ema": ema.ema.state_dict(), "ema_n": ema.n, "opt": opt.state_dict(),
                    "epoch": ep, "best": best}, os.path.join(args.out, "ckpt.pth"))
        if args.max_hours and time.time() - t_start > args.max_hours * 3600:
            print("time budget reached", flush=True)
            break
    print(f"done. best AP50 {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
