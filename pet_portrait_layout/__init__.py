"""Nodes for pastel pet portraits.

Pet Portrait Layout: places a masked subject on a canvas with a fixed amount of empty space
above it, the chest running past the bottom edge, and the subject's bounding box centred.
Pet Head Portrait Layout: the same for an untouched photo, framed by the animal's head box.
Pet Restore Colors: brushwork from a painting, colours from the photo it was painted from.
Pet Flatten Background: exact flat background that keeps the painted fur along the outline.
Pet Pastel Background: solid pastel canvas plus its name for the prompt.
Experimental: Pet Portrait Layout (padded) - space below the bust and a brushy fade at its bottom;
Pet Color Palette - recolours the animal with a gradient-map palette;
Pet Pastel Palette - 24 pastel background tones, chosen by hand, at random or to suit the coat;
Pet Split Animals - one crop per animal, so a photo of several pets gives several portraits;
Pet Face Mask - weaker painterly pass on the face, so the eyes and nose keep their likeness;
Pet Group Layout - all animals from one photo side by side on one image, or separate portraits."""
import colorsys
import json
import math
import re

import torch
import torch.nn.functional as F


class PetPortraitLayout:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "canvas": ("IMAGE", {"tooltip": "Background the subject is placed on; sets the output size."}),
            "top_space": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 0.8, "step": 0.01,
                                    "tooltip": "Share of the canvas height left empty above the subject."}),
            "bottom_overflow": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.3, "step": 0.01,
                                          "tooltip": "How far the subject runs past the bottom edge, as a share of the canvas height."}),
            "max_width": ("FLOAT", {"default": 0.9, "min": 0.2, "max": 1.0, "step": 0.01,
                                    "tooltip": "Largest allowed subject width as a share of the canvas width."}),
            "edge_softness": ("INT", {"default": 4, "min": 0, "max": 32,
                                      "tooltip": "Soft edge around the subject, in canvas pixels; 0 keeps the hard cut-out."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "layout"
    CATEGORY = "image/transform"

    def layout(self, image, mask, canvas, top_space, bottom_overflow, max_width, edge_softness):
        out = []
        for b in range(image.shape[0]):
            img = image[b]
            m = mask[min(b, mask.shape[0] - 1)].to(img)
            if m.shape != img.shape[:2]:
                m = F.interpolate(m[None, None], size=img.shape[:2], mode="bilinear")[0, 0]
            cv = canvas[min(b, canvas.shape[0] - 1)].to(img).clone()
            ch, cw = cv.shape[:2]

            ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
            if len(ys) == 0:
                out.append(cv)
                continue
            y0, y1 = ys.min().item(), ys.max().item() + 1
            x0, x1 = xs.min().item(), xs.max().item() + 1

            top = top_space * ch
            bottom = ch * (1 + bottom_overflow)
            s = min((bottom - top) / (y1 - y0), max_width * cw / (x1 - x0))
            out.append(_paste(cv, img, m, s, round(cw / 2 - (x0 + x1) / 2 * s), round(bottom - y1 * s), edge_softness))
        return (torch.stack(out),)


class PetPortraitLayoutPadded:
    """Like Pet Portrait Layout, but the bust ends inside the frame: there is empty space below it too,
    and its bottom dissolves into the background in uneven, brush-like strokes instead of a straight cut."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "canvas": ("IMAGE", {"tooltip": "Background the subject is placed on; sets the output size."}),
            "top_space": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.8, "step": 0.01,
                                    "tooltip": "Share of the canvas height left empty above the subject."}),
            "bottom_padding": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.4, "step": 0.01,
                                         "tooltip": "Empty space below the subject, as a share of the canvas height."}),
            "bottom_fade": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.8, "step": 0.01,
                                      "tooltip": "Lower share of the subject that fades into the background."}),
            "max_width": ("FLOAT", {"default": 0.86, "min": 0.2, "max": 1.0, "step": 0.01,
                                    "tooltip": "Largest allowed subject width as a share of the canvas width."}),
            "edge_softness": ("INT", {"default": 4, "min": 0, "max": 32,
                                      "tooltip": "Soft edge around the subject, in canvas pixels; 0 keeps the hard cut-out."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1, "tooltip": "Shape of the brushy fade line."}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "layout"
    CATEGORY = "image/transform"

    def layout(self, image, mask, canvas, top_space, bottom_padding, bottom_fade, max_width, edge_softness, seed):
        img = image[0]
        m = mask[0].to(img)
        if m.shape != img.shape[:2]:
            m = F.interpolate(m[None, None], size=img.shape[:2], mode="bilinear")[0, 0]
        cv = canvas[0].to(img).clone()
        ch, cw = cv.shape[:2]

        ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
        if len(ys) == 0:
            return (cv[None], torch.zeros(1, ch, cw))
        y0, y1 = ys.min().item(), ys.max().item() + 1
        x0, x1 = xs.min().item(), xs.max().item() + 1

        if bottom_fade > 0:
            m = m * _brush_fade(m.shape, y0, y1, bottom_fade, seed).to(m)

        top = top_space * ch
        bottom = ch * (1 - bottom_padding)
        s = min((bottom - top) / (y1 - y0), max_width * cw / (x1 - x0))
        ox, oy = round(cw / 2 - (x0 + x1) / 2 * s), round(bottom - y1 * s)
        out = _paste(cv, img, m, s, ox, oy, edge_softness)
        alpha = _paste(torch.zeros(ch, cw, 1).to(img), torch.ones_like(img[..., :1]), m, s, ox, oy, edge_softness)
        return (out[None], alpha[None, ..., 0])


GROUP = "group (all on one image)"
SEPARATE = "separate portraits"


class PetGroupLayout:
    """Takes the list of painted animals (one per Pet Split Animals crop). "separate" lays each out
    on its own canvas like Pet Portrait Layout (padded); "group" puts them all side by side on one
    wider canvas, equal height, the middle ones in front, each bust fading out at the bottom."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "canvas": ("IMAGE", {"tooltip": "Background; the group canvas takes the first one's colour and height."}),
            "mode": ([GROUP, SEPARATE],),
            "top_space": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.8, "step": 0.01}),
            "bottom_padding": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.4, "step": 0.01}),
            "bottom_fade": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.8, "step": 0.01}),
            "max_width": ("FLOAT", {"default": 0.86, "min": 0.2, "max": 1.0, "step": 0.01,
                                    "tooltip": "Share of the canvas width the animals may take."}),
            "overlap": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.6, "step": 0.01,
                                  "tooltip": "How much neighbours overlap, as a share of their width (group only)."}),
            "max_aspect": ("FLOAT", {"default": 1.6, "min": 0.5, "max": 3.0, "step": 0.05,
                                     "tooltip": "Widest group canvas, as width / height."}),
            "edge_softness": ("INT", {"default": 4, "min": 0, "max": 32}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
        }, "optional": {
            "head_box": ("STRING", {"forceInput": True,
                                    "tooltip": "Qwen3-VL head box of each painted animal; with head_height > 0 the "
                                               "layout is built from the head, and the bust always reaches the bottom."}),
            "head_height": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.8, "step": 0.01,
                                      "tooltip": "Head height (ears to chin) as a share of the canvas height; 0 = old layout."}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "mask", "canvas")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "layout"
    CATEGORY = "image/transform"

    def layout(self, image, mask, canvas, mode, top_space, bottom_padding, bottom_fade, max_width, overlap,
               max_aspect, edge_softness, seed, head_box=None, head_height=None):
        mode, top_space, bottom_padding, bottom_fade = mode[0], top_space[0], bottom_padding[0], bottom_fade[0]
        max_width, overlap, max_aspect, edge_softness, seed = max_width[0], overlap[0], max_aspect[0], edge_softness[0], seed[0]
        n = len(image)
        canvases = [canvas[min(i, len(canvas) - 1)] for i in range(n)]
        hh = head_height[0] if head_height else 0.0
        if hh > 0 and head_box:
            return _head_layout(image, mask, canvases, head_box, mode, top_space, bottom_padding, bottom_fade,
                                max_width, overlap, max_aspect, edge_softness, seed, hh)
        if mode == SEPARATE or n == 1:
            single = PetPortraitLayoutPadded()
            outs = [single.layout(image[i], mask[min(i, len(mask) - 1)], canvases[i], top_space, bottom_padding,
                                  bottom_fade, max_width, edge_softness, seed + i) for i in range(n)]
            return ([o[0] for o in outs], [o[1] for o in outs], canvases)

        ch = canvas[0].shape[1]
        top, bottom = top_space * ch, ch * (1 - bottom_padding)
        parts = []
        for i in range(n):
            img = image[i][0]
            m = mask[min(i, len(mask) - 1)][0].to(img)
            if m.shape != img.shape[:2]:
                m = F.interpolate(m[None, None], size=img.shape[:2], mode="bilinear")[0, 0]
            ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
            if len(ys) == 0:
                continue
            y0, y1, x0, x1 = ys.min().item(), ys.max().item() + 1, xs.min().item(), xs.max().item() + 1
            if bottom_fade > 0:
                m = m * _brush_fade(m.shape, y0, y1, bottom_fade, seed + i).to(m)
            parts.append([img, m, y0, y1, x0, x1, (bottom - top) / (y1 - y0)])
        widths = [(p[5] - p[4]) * p[6] for p in parts]
        total = sum(widths) - overlap * sum(widths[1:])
        # wider canvas for a group, rounded to 64 px for the latent; shrink the animals if still too wide
        cw = min(max(canvas[0].shape[2], total / max_width), ch * max_aspect)
        cw = max(64, int(round(cw / 64)) * 64)
        k = min(1.0, max_width * cw / total)
        pixel = canvas[0][0, 0, 0].to(parts[0][0])
        cv = pixel.expand(ch, cw, 3).clone()
        alpha = torch.zeros(ch, cw, 1).to(cv)
        x = (cw - total * k) / 2
        placed = []
        for p, wd in zip(parts, widths):
            placed.append((p, x))
            x += wd * k * (1 - overlap)
        # outer animals first, so the middle ones overlap them
        mid = (len(placed) - 1) / 2
        order = sorted(range(len(placed)), key=lambda i: -abs(i - mid))
        for (img, m, y0, y1, x0, x1, s), px in (placed[i] for i in order):
            s *= k
            ox, oy = round(px - x0 * s), round(bottom - y1 * s)
            cv = _paste(cv, img, m, s, ox, oy, edge_softness)
            alpha = torch.maximum(alpha, _paste(torch.zeros(ch, cw, 1).to(cv), torch.ones_like(img[..., :1]), m, s, ox, oy, edge_softness))
        return ([cv[None]], [alpha[None, ..., 0]], [pixel.expand(1, ch, cw, 3).clone()])


def _boxes(text, w, h):
    """Every [x1, y1, x2, y2] (0-1000 coordinates) in a Qwen3-VL answer, in pixels."""
    out = []
    for g in re.findall(r"\[([^\[\]]*)\]", text or ""):
        nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", g)[:4]]
        if len(nums) == 4 and nums[2] > nums[0] and nums[3] > nums[1]:
            out.append((nums[0] * w / 1000, nums[1] * h / 1000, nums[2] * w / 1000, nums[3] * h / 1000))
    return out


def _bbox(m):
    ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
    if len(ys) == 0:
        return None
    return ys.min().item(), ys.max().item() + 1, xs.min().item(), xs.max().item() + 1


def _head_layout(image, mask, canvases, head_box, mode, top_space, bottom_padding, bottom_fade,
                 max_width, overlap, max_aspect, edge_softness, seed, head_height):
    """Layout built from the head: every head is head_height of the canvas tall, its top at top_space;
    if the painted bust is too short to reach the bottom, the animal moves down instead (more space
    above, never a floating head). The brushy fade ends at the bottom padding line."""
    ch, cw0 = canvases[0].shape[1:3]
    top, bottom = top_space * ch, ch * (1 - bottom_padding)
    parts = []
    for i in range(len(image)):
        img = image[i][0]
        m = _fit_mask(mask[min(i, len(mask) - 1)], img.shape[:2]).to(img)
        bb = _bbox(m)
        if bb is None:
            continue
        y0, y1, x0, x1 = bb
        boxes = _boxes(head_box[min(i, len(head_box) - 1)], img.shape[1], img.shape[0])
        box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if boxes else None
        # a "head" box that covers most of the bust is really the whole animal: do not trust it
        if box and box[3] - box[1] <= 0.7 * (y1 - y0):
            hx0, hy0, hx1, hy1 = box
        else:  # estimate: the top of the animal, 0.55 of the bust tall
            hx0, hx1, hy0 = x0, x1, y0
            hy1 = y0 + 0.55 * (y1 - y0)
        parts.append({"img": img, "m": m, "y0": y0, "y1": y1, "hx": (hx0 + hx1) / 2, "hy0": hy0,
                      "s": head_height * ch / max(1.0, hy1 - hy0), "i": i})
    if not parts:
        return (canvases, [torch.zeros(1, ch, cw0) for _ in canvases], canvases)

    def fit(p):
        s = p["s"]
        oy = max(top - p["hy0"] * s, bottom - p["y1"] * s)
        ye = int(min(p["y1"], (bottom - oy) / s + 1))
        cols = torch.nonzero((p["m"][p["y0"]:max(ye, p["y0"] + 1)] > 0.5).any(0), as_tuple=True)[0]
        p.update(oy=oy, ye=ye, vx0=cols.min().item(), vx1=cols.max().item() + 1)

    def paste(cv, alpha, p, ox):
        m = p["m"]
        if bottom_fade > 0:
            m = m * _brush_fade(m.shape, p["y0"], p["ye"], bottom_fade, seed + p["i"]).to(m)
        ox, oy = round(ox), round(p["oy"])
        cv = _paste(cv, p["img"], m, p["s"], ox, oy, edge_softness)
        a = _paste(torch.zeros(cv.shape[0], cv.shape[1], 1).to(cv), torch.ones_like(p["img"][..., :1]), m, p["s"], ox, oy, edge_softness)
        return cv, torch.maximum(alpha, a)

    for p in parts:
        fit(p)
        # never wider than max_width (the visible part of the bust)
        for _ in range(2):
            wv = (p["vx1"] - p["vx0"]) * p["s"]
            if wv <= max_width * cw0:
                break
            p["s"] *= max_width * cw0 / wv
            fit(p)

    if mode == SEPARATE or len(parts) == 1:
        imgs, alphas, cvs = [], [], []
        for p in parts:
            cv = canvases[p["i"]].to(p["img"])[0].clone()
            # centre between the head and the visible bust, so the head stays near the middle
            cx = (p["hx"] + (p["vx0"] + p["vx1"]) / 2) / 2
            cv, a = paste(cv, torch.zeros(ch, cw0, 1).to(cv), p, cw0 / 2 - cx * p["s"])
            imgs.append(cv[None]); alphas.append(a[None, ..., 0]); cvs.append(canvases[p["i"]])
        return (imgs, alphas, cvs)

    widths = [(p["vx1"] - p["vx0"]) * p["s"] for p in parts]
    total = sum(widths) - overlap * sum(widths[1:])
    cw = min(max(cw0, total / max_width), ch * max_aspect)
    cw = max(64, int(round(cw / 64)) * 64)
    k = min(1.0, max_width * cw / total)
    if k < 1:
        for p in parts:
            p["s"] *= k
            fit(p)
    pixel = canvases[0][0, 0, 0].to(parts[0]["img"])
    cv = pixel.expand(ch, cw, 3).clone()
    alpha = torch.zeros(ch, cw, 1).to(cv)
    x = (cw - total * k) / 2
    placed = []
    for p, wd in zip(parts, widths):
        placed.append((p, x - p["vx0"] * p["s"]))
        x += wd * k * (1 - overlap)
    mid = (len(placed) - 1) / 2
    for j in sorted(range(len(placed)), key=lambda i: -abs(i - mid)):
        cv, alpha = paste(cv, alpha, *placed[j])
    return ([cv[None]], [alpha[None, ..., 0]], [pixel.expand(1, ch, cw, 3).clone()])


def _luma(x):
    return x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114


def _local_ratio(l, k):
    """Brightness relative to the neighbourhood: < 1 on small dark spots."""
    k = k | 1
    base = F.avg_pool2d(F.max_pool2d(l[None, None], 3, 1, 1), k, 1, k // 2, count_include_pad=False)[0, 0]
    return l / (base + 1e-3)


class PetTransferSpots:
    """Small dark spots on the nose and lips (freckles) are too small for the painter and get lost.
    This finds the nose on the photo and on the portrait (Qwen3-VL boxes), lines the photo up with the
    portrait, and darkens the portrait where the photo has a small dark spot that the portrait lacks."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "painting": ("IMAGE",),
            "painting_noses": ("STRING", {"forceInput": True, "tooltip": "Nose boxes on the portrait (one per animal)."}),
            "photo": ("IMAGE", {"tooltip": "Photo crops, one per animal, in the same order as in the portrait."}),
            "photo_nose": ("STRING", {"forceInput": True, "tooltip": "Nose box on each photo crop."}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05}),
            "darkness": ("FLOAT", {"default": 0.72, "min": 0.3, "max": 0.95, "step": 0.01,
                                   "tooltip": "A spot is at most this bright relative to its surroundings."}),
            "max_spot": ("FLOAT", {"default": 0.06, "min": 0.001, "max": 0.2, "step": 0.001,
                                   "tooltip": "Largest spot, as a share of the nose area (bigger dark areas are features, not freckles)."}),
            "pad": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "transfer"
    CATEGORY = "image/postprocessing"

    def transfer(self, painting, painting_noses, photo, photo_nose, strength, darkness, max_spot, pad):
        strength, darkness, max_spot, pad = strength[0], darkness[0], max_spot[0], pad[0]
        outs = [p.clone() for p in painting]
        jobs = []
        if len(painting) == len(photo):
            for i in range(len(painting)):
                bs = _boxes(painting_noses[min(i, len(painting_noses) - 1)], painting[i].shape[2], painting[i].shape[1])
                if bs:
                    jobs.append((i, max(bs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])), i))
        else:  # one group portrait: noses left to right, photos in the same order
            bs = sorted(_boxes(painting_noses[0], painting[0].shape[2], painting[0].shape[1]), key=lambda b: b[0] + b[2])
            jobs = [(0, b, j) for j, b in enumerate(bs[:len(photo)])]
        for pi, pbox, qi in jobs:
            q = photo[qi][0]
            qb = _boxes(photo_nose[min(qi, len(photo_nose) - 1)], q.shape[1], q.shape[0])
            if qb:
                outs[pi][0] = self._one(outs[pi][0], pbox, q, max(qb, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])),
                                        strength, darkness, max_spot, pad)
        return (outs,)

    @staticmethod
    def _crop(img, b, pad):
        h, w = img.shape[:2]
        px, py = (b[2] - b[0]) * pad, (b[3] - b[1]) * pad
        x0, y0 = max(0, int(b[0] - px)), max(0, int(b[1] - py))
        x1, y1 = min(w, int(b[2] + px + 0.5)), min(h, int(b[3] + py + 0.5))
        return x0, y0, x1, y1

    def _one(self, P, pbox, Q, qbox, strength, darkness, max_spot, pad):
        import scipy.ndimage as ndi
        px0, py0, px1, py1 = self._crop(P, pbox, pad)
        qx0, qy0, qx1, qy1 = self._crop(Q, qbox, pad)
        if px1 - px0 < 16 or py1 - py0 < 16 or qx1 - qx0 < 8 or qy1 - qy0 < 8:
            return P
        pc = P[py0:py1, px0:px1]
        hc, wc = pc.shape[:2]
        qc = F.interpolate(Q[qy0:qy1, qx0:qx1].movedim(-1, 0)[None], size=(hc, wc), mode="bicubic",
                           antialias=True).clamp(0, 1)

        # line the photo up with the portrait: brute-force scale and shift on edge maps
        def edges(x):  # x: 1x3xhxw
            l = _luma(x.movedim(1, -1))[:, None]
            gx = l[..., :, 1:] - l[..., :, :-1]
            gy = l[..., 1:, :] - l[..., :-1, :]
            g = gx[..., :-1, :].abs() + gy[..., :, :-1].abs()
            return F.avg_pool2d(g, 3, 1, 1)

        sc_lo = 96 / max(hc, wc)
        lo = (max(16, round(hc * sc_lo)), max(16, round(wc * sc_lo)))
        pe = edges(F.interpolate(pc.movedim(-1, 0)[None], size=lo, mode="area"))
        pe = (pe - pe.mean()) / (pe.std() + 1e-6)
        best, theta_best = -1e9, None
        shifts = torch.linspace(-0.24, 0.24, 25)
        for s in (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15):
            ty, tx = torch.meshgrid(shifts, shifts, indexing="ij")
            n = tx.numel()
            theta = torch.zeros(n, 2, 3)
            theta[:, 0, 0] = 1 / s
            theta[:, 1, 1] = 1 / s
            theta[:, 0, 2] = tx.flatten()
            theta[:, 1, 2] = ty.flatten()
            grid = F.affine_grid(theta, (n, 3) + lo, align_corners=False)
            qlo = F.interpolate(qc, size=lo, mode="area").expand(n, -1, -1, -1)
            qe = edges(F.grid_sample(qlo, grid, padding_mode="border", align_corners=False))
            qe = (qe - qe.mean((1, 2, 3), keepdim=True)) / (qe.std((1, 2, 3), keepdim=True) + 1e-6)
            score = (qe * pe).mean((1, 2, 3))
            j = int(score.argmax())
            if score[j] > best:
                best, theta_best = float(score[j]), theta[j:j + 1]
        if best < 0.15:  # the noses do not match: leave the portrait alone
            return P
        grid = F.affine_grid(theta_best, (1, 3, hc, wc), align_corners=False)
        qa = F.grid_sample(qc, grid, padding_mode="border", align_corners=False)[0].movedim(0, -1)

        k = max(5, round(min(hc, wc) / 5))
        rq, rp = _local_ratio(_luma(qa), k), _local_ratio(_luma(pc), k)
        # only on bare pink skin (nose leather, lips): the neighbourhood is pinkish, not orange or white fur
        kk = max(3, k // 3) | 1
        clean = F.max_pool2d(qa.movedim(-1, 0)[None], 5, 1, 2)  # spots removed, the skin colour remains
        near = F.avg_pool2d(clean, kk, 1, kk // 2, count_include_pad=False)[0].movedim(0, -1)
        r_, g_, b_ = near[..., 0], near[..., 1], near[..., 2]
        skin = (r_ - g_ > 0.05) & (b_ - g_ > -0.06)
        skin = F.max_pool2d(skin.float()[None, None], 7, 1, 3)[0, 0] > 0
        cand = (rq < darkness) & (rq < rp - 0.12) & (_luma(qa) < 0.5) & skin
        lab, nlab = ndi.label(cand.cpu().numpy())
        keep = torch.zeros(hc, wc)
        if nlab:
            limit = max_spot * hc * wc / (1 + 2 * pad) ** 2
            for idx, sl in enumerate(ndi.find_objects(lab), start=1):
                comp = lab[sl] == idx
                sz = comp.sum()
                bh, bw = comp.shape
                # freckles are compact blobs; long thin strokes are whiskers or fur
                if 2 <= sz <= limit and max(bh, bw) <= 3 * max(1, min(bh, bw)) and sz >= 0.25 * bh * bw:
                    keep[sl][torch.from_numpy(comp)] = 1
        if keep.sum() == 0:
            return P
        a = F.avg_pool2d(keep[None, None], 3, 1, 1)[0, 0].to(pc)[..., None] * strength
        f = (rq / rp.clamp(min=1e-3)).clamp(0.15, 1.0)[..., None]
        P = P.clone()
        P[py0:py1, px0:px1] = (pc * (1 - a.clamp(0, 1) * (1 - f))).clamp(0, 1)
        return P


def _pair_boxes(painting, painting_boxes, photo):
    """(painting index, box on it, photo index): one painting per photo, or one group painting whose
    boxes, left to right, follow the photos."""
    jobs = []
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    if len(painting) == len(photo):
        for i in range(len(painting)):
            bs = _boxes(painting_boxes[min(i, len(painting_boxes) - 1)], painting[i].shape[2], painting[i].shape[1])
            if bs:
                jobs.append((i, max(bs, key=area), i))
    else:
        bs = sorted(_boxes(painting_boxes[0], painting[0].shape[2], painting[0].shape[1]), key=lambda b: b[0] + b[2])
        jobs = [(0, b, j) for j, b in enumerate(bs[:len(photo)])]
    return jobs


def _edges(x):
    """Edge map of a batch of images (n x 3 x h x w) for alignment."""
    l = _luma(x.movedim(1, -1))[:, None]
    gx = (l[..., :, 1:] - l[..., :, :-1])[..., :-1, :]
    gy = (l[..., 1:, :] - l[..., :-1, :])[..., :, :-1]
    return F.avg_pool2d(gx.abs() + gy.abs(), 3, 1, 1)


def _align(src, dst, scales, angles, shift=0.24, steps=17):
    """Brute-force similarity transform (scale, rotation, shift) that lines src up with dst
    (both 1 x 3 x h x w, same size). Returns (theta, score)."""
    import math
    h, w = dst.shape[2:]
    k = 96 / max(h, w)
    lo = (max(16, round(h * k)), max(16, round(w * k)))
    de = _edges(F.interpolate(dst, size=lo, mode="area"))
    de = (de - de.mean()) / (de.std() + 1e-6)
    slo = F.interpolate(src, size=lo, mode="area")
    sh = torch.linspace(-shift, shift, steps)
    ty, tx = torch.meshgrid(sh, sh, indexing="ij")
    n = tx.numel()
    best, best_t = -1e9, None
    for a in angles:
        ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
        for s in scales:
            th = torch.zeros(n, 2, 3)
            th[:, 0, 0], th[:, 0, 1] = ca / s, -sa / s
            th[:, 1, 0], th[:, 1, 1] = sa / s, ca / s
            th[:, 0, 2], th[:, 1, 2] = tx.flatten(), ty.flatten()
            grid = F.affine_grid(th, (n, 3) + lo, align_corners=False)
            se = _edges(F.grid_sample(slo.expand(n, -1, -1, -1), grid, padding_mode="border", align_corners=False))
            se = (se - se.mean((1, 2, 3), keepdim=True)) / (se.std((1, 2, 3), keepdim=True) + 1e-6)
            sc = (se * de).mean((1, 2, 3))
            j = int(sc.argmax())
            if sc[j] > best:
                best, best_t = float(sc[j]), th[j:j + 1]
    return best_t, best


def _face_points(text, w, h):
    """Two eyes and the nose from a Qwen3-VL answer with "point_2d" entries (0-1000 coordinates).
    Labels are not trusted (whose left?): the two highest points are the eyes, sorted left to right,
    the lowest one is the nose. Returns [(x, y) eye_l, eye_r, nose] in pixels, or None."""
    pts = []
    for g in re.findall(r"point_2d\"?\s*:\s*\[([^\[\]]*)\]", text or ""):
        nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", g)[:2]]
        if len(nums) == 2:
            pts.append((nums[0] * w / 1000, nums[1] * h / 1000))
    if len(pts) < 3:
        return None
    pts = sorted(pts[:3], key=lambda p: p[1])
    eyes = sorted(pts[:2], key=lambda p: p[0])
    nose = pts[2]
    d = ((eyes[1][0] - eyes[0][0]) ** 2 + (eyes[1][1] - eyes[0][1]) ** 2) ** 0.5
    if d < 4 or nose[1] - (eyes[0][1] + eyes[1][1]) / 2 < 0.15 * d:
        return None
    return [eyes[0], eyes[1], nose]


def _pose(pts):
    (lx, ly), (rx, ry), (nx, ny) = pts
    ex, ey = rx - lx, ry - ly
    d2 = ex * ex + ey * ey or 1.0
    mx, my = (lx + rx) / 2, (ly + ry) / 2
    vx, vy = nx - mx, ny - my
    yaw = (vx * ex + vy * ey) / d2            # sideways nose shift / eye distance
    drop = (vx * -ey + vy * ex) / d2          # nose depth below eyes / eye distance
    return yaw, drop


def _same_pose(pp, qp, max_yaw=0.12, max_drop=0.25):
    a, b = _pose(pp), _pose(qp)
    return abs(a[0] - b[0]) <= max_yaw and abs(a[1] - b[1]) <= max_drop * max(abs(a[1]), 1e-3)


class PetFaceFromPhoto:
    """The painters keep the general look but lose the individual face: exact spots, stripes, the
    asymmetry of the markings. This puts the real face from the photo onto the painted animal before
    the painterly pass. Both eyes and the nose are found on the photo and on the painting (Qwen3-VL
    points), the photo is mapped onto the painting through these three points, its colours are
    brought to the painting, and the face (eyes, nose, muzzle, cheeks) is blended in with a soft
    oval. The SDXL pass after it paints over the face, so it is painted, but from the photo."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "painting": ("IMAGE", {"tooltip": "Painted animals, one per photo crop (before the layout)."}),
            "painting_points": ("STRING", {"forceInput": True, "tooltip": "Eyes and nose on each painting (point_2d)."}),
            "photo": ("IMAGE", {"tooltip": "Photo crops, one per animal, in the same order."}),
            "photo_points": ("STRING", {"forceInput": True, "tooltip": "Eyes and nose on each photo crop (point_2d)."}),
            "strength": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "How much of the photo face goes in (1 = all of it inside the oval)."}),
            "size": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                               "tooltip": "Size of the oval: 1 covers eyes, nose, muzzle and cheeks."}),
            "feather": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 1.0, "step": 0.05,
                                  "tooltip": "Soft edge of the oval, as a share of its size."}),
            "color_match": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "Bring the photo face's overall colour and contrast to the painting's."}),
        }, "optional": {
            "mask": ("MASK", {"tooltip": "Animal mask of each painting: the photo never spills onto the background."}),
            "photo_mask": ("MASK", {"tooltip": "Animal mask of each photo crop: the photo background is never taken."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "apply"
    CATEGORY = "image/postprocessing"

    def apply(self, painting, painting_points, photo, photo_points, strength, size, feather, color_match, mask=None,
              photo_mask=None):
        self.photo_mask = photo_mask
        strength, size, feather, color_match = strength[0], size[0], feather[0], color_match[0]
        outs = []
        for i in range(len(painting)):
            P = painting[i]
            if i >= len(photo):
                outs.append(P); continue
            Q = photo[i][0]
            pp = _face_points(painting_points[min(i, len(painting_points) - 1)], P.shape[2], P.shape[1])
            qp = _face_points(photo_points[min(i, len(photo_points) - 1)], Q.shape[1], Q.shape[0])
            m = mask[min(i, len(mask) - 1)] if mask else None
            if pp is None or qp is None or len(painting) > 1 or not _same_pose(pp, qp):
                print("[PetFaceFromPhoto] skip item", i, "(no points, group or pose mismatch)")
                outs.append(P); continue
            self.qm = self.photo_mask[min(i, len(self.photo_mask) - 1)] if self.photo_mask else None
            outs.append(self._one(P[0], pp, Q, qp, m, strength, size, feather, color_match)[None])
        return (outs,)

    def _one(self, P, pp, Q, qp, m, strength, size, feather, color_match):
        H, W = P.shape[:2]
        Qh, Qw = Q.shape[:2]
        # affine map painting pixel -> photo pixel through the three point pairs
        A = torch.tensor([[x, y, 1.0] for x, y in pp], dtype=torch.float64)
        B = torch.tensor(qp, dtype=torch.float64)
        try:
            M = torch.linalg.solve(A, B).T  # 2 x 3
        except RuntimeError:
            return P
        # oval around the face on the painting
        (lx, ly), (rx_, ry_), (nx, ny) = pp
        d = ((rx_ - lx) ** 2 + (ry_ - ly) ** 2) ** 0.5
        ex, ey = (lx + rx_) / 2, (ly + ry_) / 2
        cx, cy = ex + 0.4 * (nx - ex), ey + 0.4 * (ny - ey)
        ra, rb = 1.05 * d * size, 1.2 * d * size
        x0, x1 = int(max(0, cx - ra * 1.2)), int(min(W, cx + ra * 1.2))
        y0, y1 = int(max(0, cy - rb * 1.2)), int(min(H, cy + rb * 1.2))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return P
        ys = torch.arange(y0, y1, dtype=torch.float64)[:, None].expand(-1, x1 - x0)
        xs = torch.arange(x0, x1, dtype=torch.float64)[None, :].expand(y1 - y0, -1)
        qx = M[0, 0] * xs + M[0, 1] * ys + M[0, 2]
        qy = M[1, 0] * xs + M[1, 1] * ys + M[1, 2]
        grid = torch.stack([(qx + 0.5) / Qw * 2 - 1, (qy + 0.5) / Qh * 2 - 1], -1)[None].to(Q)
        qa = F.grid_sample(Q.movedim(-1, 0)[None], grid, mode="bicubic", padding_mode="border",
                           align_corners=False)[0].movedim(0, -1).clamp(0, 1)
        valid = ((grid[0, ..., 0].abs() <= 1) & (grid[0, ..., 1].abs() <= 1)).to(P)

        r = torch.sqrt(((xs - cx) / ra) ** 2 + ((ys - cy) / rb) ** 2).to(P)
        a = ((1 - r) / feather + 0.5).clamp(0, 1)
        a = a * a * (3 - 2 * a) * valid
        if getattr(self, "qm", None) is not None:
            qmask = _fit_mask(self.qm, (Qh, Qw)).to(Q)
            qmask = -F.max_pool2d(-qmask[None, None], 9, 1, 4)  # shrink: keep off the photo outline
            a = a * F.grid_sample(qmask, grid, padding_mode="zeros", align_corners=False)[0, 0].to(P)
        if m is not None:
            mm = _fit_mask(m, (H, W)).to(P)[y0:y1, x0:x1]
            a = a * F.avg_pool2d(mm[None, None], 9, 1, 4)[0, 0]
        if a.sum() < 16:
            return P
        pc = P[y0:y1, x0:x1]
        w8 = a[..., None]
        tot = w8.sum()
        mq, mp = (qa * w8).sum((0, 1)) / tot, (pc * w8).sum((0, 1)) / tot
        sq = (((qa - mq) ** 2 * w8).sum((0, 1)) / tot).sqrt() + 1e-4
        sp = (((pc - mp) ** 2 * w8).sum((0, 1)) / tot).sqrt() + 1e-4
        qa = (qa + ((qa - mq) / sq * sp + mp - qa) * color_match).clamp(0, 1)
        P = P.clone()
        k = (a * strength)[..., None]
        P[y0:y1, x0:x1] = qa * k + pc * (1 - k)
        return P


def _brush_fade(shape, y0, y1, fade, seed):
    """Alpha that goes from 1 to 0 over the lowest `fade` share of rows y0..y1, with a ragged,
    stroke-like end line: each column fades at a slightly different height."""
    h, w = shape
    g = torch.Generator().manual_seed(seed)

    def noise(knots):
        v = torch.rand(1, 1, knots, generator=g) * 2 - 1
        return F.interpolate(v, size=w, mode="linear", align_corners=True)[0, 0]

    jitter = 0.3 * noise(9) + 0.15 * noise(70)
    length = max(1.0, fade * (y1 - y0))
    rows = torch.arange(h, dtype=torch.float32)[:, None]
    t = (rows - (y1 - length)) / length + jitter[None, :]
    t = t.clamp(0, 1)
    return 1 - t * t * (3 - 2 * t)


def _soften(m, r):
    """Shrink the mask by r px, then blur it by r px: a soft edge that stays inside the old outline,
    so no fringe of the original background is blended in."""
    k = 2 * r + 1
    m = -F.max_pool2d(-m[None, None], k, 1, r)
    for _ in range(2):
        m = F.avg_pool2d(m, k, 1, r, count_include_pad=False)
    return m[0, 0]


def _paste(cv, img, m, s, ox, oy, softness=0):
    """Scale img/mask by s and alpha-blend them onto canvas cv with the top-left corner at (ox, oy)."""
    ch, cw = cv.shape[:2]
    nh, nw = max(1, round(img.shape[0] * s)), max(1, round(img.shape[1] * s))
    simg = F.interpolate(img.movedim(-1, 0)[None], size=(nh, nw), mode="bicubic", antialias=True)[0].movedim(0, -1).clamp(0, 1)
    sm = F.interpolate(m[None, None], size=(nh, nw), mode="bilinear")[0, 0].clamp(0, 1)
    if softness > 0:
        sm = _soften(sm, softness)
    ty0, tx0 = max(oy, 0), max(ox, 0)
    ty1, tx1 = min(oy + nh, ch), min(ox + nw, cw)
    if ty1 > ty0 and tx1 > tx0:
        src = simg[ty0 - oy:ty1 - oy, tx0 - ox:tx1 - ox]
        a = sm[ty0 - oy:ty1 - oy, tx0 - ox:tx1 - ox, None]
        cv[ty0:ty1, tx0:tx1] = src * a + cv[ty0:ty1, tx0:tx1] * (1 - a)
    return cv


class PetHeadPortraitLayout:
    """Places the cut-out animal from a photo as a portrait, keeping its original pose: the head box
    (Qwen3-VL "bbox_2d" in 0-1000 coordinates) is centred horizontally, its top sits at top_space and
    the body runs past the bottom edge."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "head_box": ("STRING", {"forceInput": True, "tooltip": 'Text containing [x1, y1, x2, y2] in 0-1000 coordinates.'}),
            "canvas": ("IMAGE", {"tooltip": "Background the animal is placed on; sets the output size."}),
            "top_space": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.7, "step": 0.01,
                                    "tooltip": "Share of the canvas height left empty above the head."}),
            "head_height": ("FLOAT", {"default": 0.34, "min": 0.1, "max": 0.8, "step": 0.01,
                                      "tooltip": "Head height (ears to chin) as a share of the canvas height."}),
            "max_head_width": ("FLOAT", {"default": 0.7, "min": 0.2, "max": 1.0, "step": 0.01}),
            "bottom_overflow": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.3, "step": 0.01}),
            "edge_fade": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.2, "step": 0.01,
                                    "tooltip": "Fade the animal out where the photo frame cuts it on the left, right or top."}),
            "edge_softness": ("INT", {"default": 4, "min": 0, "max": 32,
                                      "tooltip": "Soft edge around the animal, in canvas pixels; 0 keeps the hard cut-out."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "layout"
    CATEGORY = "image/transform"

    def layout(self, image, mask, head_box, canvas, top_space, head_height, max_head_width, bottom_overflow, edge_fade, edge_softness):
        img = image[0]
        h, w = img.shape[:2]
        m = mask[0].to(img)
        if m.shape != img.shape[:2]:
            m = F.interpolate(m[None, None], size=(h, w), mode="bilinear")[0, 0]
        cv = canvas[0].to(img).clone()
        ch, cw = cv.shape[:2]

        ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
        if len(ys) == 0:
            return (cv[None],)
        body_bottom = ys.max().item() + 1

        box = re.search(r"\[([^\]]*)\]", head_box)  # "bbox_2d" itself contains a digit
        nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", box.group(1))[:4]] if box else []
        if len(nums) == 4 and nums[2] > nums[0] and nums[3] > nums[1]:
            hx0, hy0, hx1, hy1 = nums[0] * w / 1000, nums[1] * h / 1000, nums[2] * w / 1000, nums[3] * h / 1000
        else:
            # No usable head box: take the top of the animal, as tall as it is wide.
            x0, x1, y0 = xs.min().item(), xs.max().item() + 1, ys.min().item()
            hx0, hx1, hy0 = x0, x1, y0
            hy1 = min(y0 + (x1 - x0) * 0.8, body_bottom)

        if edge_fade > 0:
            r = max(1, round(edge_fade * min(h, w)))
            ramp = torch.linspace(0, 1, r, device=m.device, dtype=m.dtype)
            m = m.clone()
            m[:r] *= ramp[:, None]
            m[:, :r] *= ramp[None, :]
            m[:, -r:] *= ramp.flip(0)[None, :]

        top = top_space * ch
        bottom = ch * (1 + bottom_overflow)
        s = head_height * ch / (hy1 - hy0)
        # When the photo ends too high the chest would float: first grow the animal (up to the
        # head width limit), then move it down, which only adds empty space above.
        s = max(s, (bottom - top) / (body_bottom - hy0))
        s = min(s, max_head_width * cw / (hx1 - hx0))
        ox = round(cw / 2 - (hx0 + hx1) / 2 * s)
        oy = round(max(top - hy0 * s, bottom - body_bottom * s))
        return (_paste(cv, img, m, s, ox, oy, edge_softness)[None],)


class PetFlattenBackground:
    """Replaces the background with the exact flat colour, but only away from the animal: the mask is
    grown and feathered so the painted wisps of fur along the outline stay and no cut-out edge appears."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "canvas": ("IMAGE", {"tooltip": "Flat background of the same size as the image."}),
            # A wide band also keeps whatever the models painted next to the animal (e.g. a shadow).
            "protect": ("INT", {"default": 4, "min": 0, "max": 128,
                                "tooltip": "Pixels around the animal kept as painted (fur wisps)."}),
            "feather": ("INT", {"default": 8, "min": 1, "max": 128,
                                "tooltip": "Width of the soft transition into the flat background, in pixels."}),
        }, "optional": {
            "extra_mask": ("MASK", {"tooltip": "Also kept as painted, e.g. the faded bottom of the bust from the layout."}),
            # SDXL likes to sign the empty corner of a vignette; capping by the layout keeps the fade clean.
            "limit": ("INT", {"default": 0, "min": 0, "max": 256,
                              "tooltip": "If > 0, nothing further than this many pixels outside extra_mask is kept, "
                                         "and its faded part stays faded."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "flatten"
    CATEGORY = "image/postprocessing"

    def flatten(self, image, mask, canvas, protect, feather, extra_mask=None, limit=0):
        m = mask.to(image)[:, None]
        if m.shape[2:] != image.shape[1:3]:
            m = F.interpolate(m, size=image.shape[1:3], mode="bilinear")
        if extra_mask is not None:
            e = F.interpolate(extra_mask.to(image)[:, None], size=image.shape[1:3], mode="bilinear")
            m = torch.maximum(m, e)
            if limit > 0:
                m = torch.minimum(m, F.max_pool2d(e, 2 * limit + 1, 1, limit))
        if protect > 0:
            m = F.max_pool2d(m, 2 * protect + 1, 1, protect)
        for _ in range(2):
            m = F.avg_pool2d(m, 2 * feather + 1, 1, feather, count_include_pad=False)
        a = m.movedim(1, -1)
        cv = canvas.to(image)
        if cv.shape[1:3] != image.shape[1:3]:
            cv = F.interpolate(cv.movedim(-1, 1), size=image.shape[1:3], mode="nearest").movedim(1, -1)
        return (image * a + cv * (1 - a),)


_TO_YCBCR = torch.tensor([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])


class PetRestoreColors:
    """Keeps the brushwork (brightness) of a painting but takes the colours from the photo it was
    painted from: the edit model tends to invent coat colours, e.g. turning a silver cat brown."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "painting": ("IMAGE",),
            "photo": ("IMAGE", {"tooltip": "The framed photo with the same layout as the painting."}),
            "color_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "photo_brightness": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                                           "tooltip": "Share of the photo's brightness mixed back in to recover fine detail."}),
            "color_blur": ("INT", {"default": 8, "min": 1, "max": 64,
                                   "tooltip": "Colours are taken from the photo shrunk by this factor, so small misalignments do not show."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "restore"
    CATEGORY = "image/postprocessing"

    def restore(self, painting, photo, color_strength, photo_brightness, color_blur):
        h, w = painting.shape[1:3]
        photo = photo.movedim(-1, 1).to(painting)
        if photo.shape[2:] != (h, w):
            photo = F.interpolate(photo, size=(h, w), mode="bilinear", antialias=True)
        soft = F.interpolate(F.interpolate(photo, size=(max(1, h // color_blur), max(1, w // color_blur)), mode="area"),
                             size=(h, w), mode="bilinear")
        m = _TO_YCBCR.to(painting)
        paint_ycc = painting @ m.T
        photo_ycc = photo.movedim(1, -1) @ m.T
        soft_ycc = soft.movedim(1, -1) @ m.T
        y = paint_ycc[..., :1] * (1 - photo_brightness) + photo_ycc[..., :1] * photo_brightness
        cbcr = paint_ycc[..., 1:] * (1 - color_strength) + soft_ycc[..., 1:] * color_strength
        out = torch.cat([y, cbcr], dim=-1) @ torch.linalg.inv(m).T
        return (out.clamp(0, 1),)


class PetMatchColors:
    """Gives a newly painted animal the coat colours of the photo. The pose differs, so pixel-wise
    colour restore is impossible; instead the colour statistics (mean and spread of the chroma, and
    optionally brightness) of the animal are matched to the photo's animal: a ginger-tinted grey cat
    turns grey again, while the brushwork and markings stay."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "painting": ("IMAGE",),
            "painting_mask": ("MASK",),
            "photo": ("IMAGE",),
            "photo_mask": ("MASK",),
            "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
            "brightness": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                                     "tooltip": "How much of the photo's overall brightness and contrast to take as well."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "match"
    CATEGORY = "image/postprocessing"

    def match(self, painting, painting_mask, photo, photo_mask, strength, brightness):
        m = _TO_YCBCR.to(painting)
        p = painting[0] @ m.T
        q = photo[0].to(painting) @ m.T
        pm = _fit_mask(painting_mask, p.shape[:2]).to(p) > 0.5
        qm = _fit_mask(photo_mask, q.shape[:2]).to(p) > 0.5
        if pm.sum() < 16 or qm.sum() < 16:
            return (painting,)
        ps, qs = p[pm], q[qm]
        mean_p, std_p = ps.mean(0), ps.std(0) + 1e-4
        mean_q, std_q = qs.mean(0), qs.std(0) + 1e-4
        matched = (p - mean_p) / std_p * std_q + mean_q
        k = torch.tensor([brightness, 1.0, 1.0]).to(p) * strength
        out = p + (matched - p) * k
        a = pm.float()[..., None]
        out = out * a + p * (1 - a)
        return ((out @ torch.linalg.inv(m).T).clamp(0, 1)[None],)


class PetCropToBox:
    """Crops the image to a Qwen3-VL box (0-1000 coordinates) with some margin and scales it to a
    square, e.g. a close-up of the face as an extra reference for the painter."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "box": ("STRING", {"forceInput": True, "tooltip": "Text containing [x1, y1, x2, y2] in 0-1000 coordinates."}),
            "pad": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
            "size": ("INT", {"default": 768, "min": 64, "max": 2048, "step": 16}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop"
    CATEGORY = "image/transform"

    def crop(self, image, box, pad, size):
        h, w = image.shape[1:3]
        found = re.search(r"\[([^\]]*)\]", box)
        nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", found.group(1))[:4]] if found else []
        if len(nums) == 4 and nums[2] > nums[0] and nums[3] > nums[1]:
            x0, y0, x1, y1 = nums[0] * w / 1000, nums[1] * h / 1000, nums[2] * w / 1000, nums[3] * h / 1000
        else:
            x0, y0, x1, y1 = 0, 0, w, h
        side = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x1, y1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
        crop = image[:1, y0:y1, x0:x1].movedim(-1, 1)
        return (F.interpolate(crop, size=(size, size), mode="bicubic", antialias=True).movedim(1, -1).clamp(0, 1),)


def _fit_mask(mask, size):
    m = mask[:1].float()
    if m.shape[1:] != tuple(size):
        m = F.interpolate(m[None], size=size, mode="bilinear")[0]
    return m[0]


# Gradient maps from shadows to highlights. The animal keeps its own brightness (all the brushwork),
# only the hues are taken from the palette, so the result reads as one colour scheme with the background.
PALETTES = {
    "original colors": None,
    "match background": None,  # built from the canvas colour
    "warm sepia": [0x3B2A20, 0x8A6A4F, 0xD9BF9C, 0xFBF3E6],
    "terracotta": [0x3A1F1A, 0x9A4E3A, 0xE0A084, 0xFBEEE4],
    "dusty rose": [0x3A2630, 0x8E5A68, 0xD9A8B0, 0xFBF0F0],
    "sage olive": [0x262B20, 0x5F6B4A, 0xB5BF9B, 0xF4F6EC],
    "blue hour": [0x1E2433, 0x4F5F80, 0xA9B6CF, 0xF2F4FA],
    "ochre charcoal": [0x262321, 0x6B5A3E, 0xD2A95A, 0xFAF2DF],
    "lavender dusk": [0x2A2433, 0x6E5F86, 0xBDB0D4, 0xF6F2FA],
}


def _rgb(c):
    return [((c >> 16) & 0xFF) / 255.0, ((c >> 8) & 0xFF) / 255.0, (c & 0xFF) / 255.0]


class PetColorPalette:
    """Recolours the animal with a palette (a gradient map from dark to light) while keeping its
    brightness, so fur detail survives; strength 0 leaves the colours untouched."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "canvas": ("IMAGE", {"tooltip": "The background; used by \"match background\"."}),
            "palette": (list(PALETTES),),
            "strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "How much of the palette's hue replaces the animal's own colours."}),
            "tone": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                               "tooltip": "How much of the palette's light and dark values is mixed in as well."}),
        }, "optional": {
            "mask": ("MASK", {"tooltip": "Only this area is recoloured (the background stays exact)."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/postprocessing"

    def apply(self, image, canvas, palette, strength, tone, mask=None):
        if palette == "original colors" or strength == 0:
            return (image,)
        if palette == "match background":
            bg = canvas[0, 0, 0].float().cpu()
            ycc = bg @ _TO_YCBCR.T
            # same hue as the background, more saturated towards the shadows
            stops = [torch.cat([y, ycc[1:] * k]) @ torch.linalg.inv(_TO_YCBCR).T
                     for y, k in ((torch.tensor([0.16]), 3.0), (torch.tensor([0.45]), 3.0), (torch.tensor([0.78]), 2.0), (torch.tensor([0.97]), 0.6))]
            stops = torch.stack(stops).clamp(0, 1)
        else:
            stops = torch.tensor([_rgb(c) for c in PALETTES[palette]])
        stops = stops.to(image)

        m = _TO_YCBCR.to(image)
        ycc = image @ m.T
        y = ycc[..., 0].clamp(0, 1)
        # piecewise-linear gradient map over the stops, evenly spaced in brightness
        pos = y * (len(stops) - 1)
        i = pos.floor().clamp(max=len(stops) - 2).long()
        f = (pos - i)[..., None]
        mapped = stops[i] * (1 - f) + stops[i + 1] * f
        mycc = mapped @ m.T
        out_y = ycc[..., :1] * (1 - tone) + mycc[..., :1] * tone
        out_c = ycc[..., 1:] * (1 - strength) + mycc[..., 1:] * strength
        out = (torch.cat([out_y, out_c], dim=-1) @ torch.linalg.inv(m).T).clamp(0, 1)
        if mask is not None:
            a = mask.to(image)[:, None]
            if a.shape[2:] != image.shape[1:3]:
                a = F.interpolate(a, size=image.shape[1:3], mode="bilinear")
            a = a.movedim(1, -1)
            out = out * a + image * (1 - a)
        return (out,)


PASTELS = {
    "pale warm cream": 0xF3E6CC,
    "soft dusty pink": 0xEBD5D0,
    "pale sage green": 0xDCE3D2,
    "soft powder blue": 0xD6E0E6,
    "warm pale apricot": 0xF2DCC8,
    "pale lavender": 0xE2DCE8,
    "light butter yellow": 0xF3EBCB,
    "soft greige": 0xE6E1DA,
}


class PetPastelBackground:
    """Solid pastel canvas plus its name, so one dropdown drives both the prompt and the fill."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "color": (list(PASTELS),),
            "width": ("INT", {"default": 896, "min": 64, "max": 4096, "step": 16}),
            "height": ("INT", {"default": 1120, "min": 64, "max": 4096, "step": 16}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("canvas", "color_name")
    FUNCTION = "make"
    CATEGORY = "image"

    def make(self, color, width, height):
        rgb = PASTELS[color]
        pixel = torch.tensor([(rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF], dtype=torch.float32) / 255.0
        return (pixel.expand(1, height, width, 3).clone(), color)


# Extended background palette, grouped; the names go into the prompt, so they describe the colour.
PASTEL_GROUPS = {
    "warm": {
        "pale warm cream": 0xF3E6CC,
        "pale vanilla": 0xF6EFD9,
        "light butter yellow": 0xF3EBCB,
        "warm pale apricot": 0xF2DCC8,
        "soft peach": 0xF5D9C6,
        "pale coral pink": 0xF4D4CC,
        "soft blush pink": 0xF2DADA,
        "soft dusty pink": 0xEBD5D0,
        "light terracotta blush": 0xE9CDBF,
    },
    "cool": {
        "pale pistachio": 0xE4EAD0,
        "pale sage green": 0xDCE3D2,
        "soft mint": 0xE0EDE4,
        "soft duck egg blue": 0xD5E5E2,
        "pale sky blue": 0xDCE8F0,
        "soft powder blue": 0xD6E0E6,
        "pale periwinkle": 0xDCDFEE,
        "pale lavender": 0xE2DCE8,
        "soft lilac": 0xE8DDEB,
    },
    "neutral": {
        "soft linen": 0xF1ECE3,
        "pale oat beige": 0xEDE6DA,
        "soft greige": 0xE6E1DA,
        "warm light stone": 0xE3DDD3,
        "pale mushroom": 0xDCD3CC,
        "soft dove grey": 0xDFDFDD,
    },
}
PASTEL_PALETTE = {n: c for g in PASTEL_GROUPS.values() for n, c in g.items()}
AUTO = "auto (по окрасу)"
RANDOM = {"random": list(PASTEL_PALETTE), "random warm": list(PASTEL_GROUPS["warm"]),
          "random cool": list(PASTEL_GROUPS["cool"]), "random neutral": list(PASTEL_GROUPS["neutral"])}


def _hue_dist(a, b):
    d = abs(a - b) % 1.0
    return min(d, 1 - d) * 360


def _auto_pastel(rgb, seed=0, top=3):
    """Background that sets off the coat: a colour roughly opposite on the colour wheel for coloured
    coats, warm pastels for grey and black ones, and deeper tones for white ones so they don't vanish.
    The seed picks one of the `top` best tones, so repeated runs vary but stay harmonious."""
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    scored = []
    for name, c in PASTEL_PALETTE.items():
        ch, cs, cv = colorsys.rgb_to_hsv(*_rgb(c))
        if cs < 0.06:
            score = 0.45  # neutral tones go with anything, but are never the most flattering
        elif s > 0.2 and v > 0.15:
            score = 1 - abs(_hue_dist(h, ch) - 150) / 150
        else:
            score = 1 - _hue_dist(ch, 20 / 360) / 180
        if v > 0.75 and s < 0.25:
            score += (1 - cv) * 3 + cs * 2
        scored.append((score, name))
    scored.sort(reverse=True)
    return scored[seed % min(top, len(scored))][1]


class PetPastelPalette:
    """Pastel background from an extended palette (24 tones in warm, cool and neutral groups).
    "auto" picks the tone that suits the animal's coat; "random ..." picks by seed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "color": ([AUTO] + list(RANDOM) + list(PASTEL_PALETTE),),
            "width": ("INT", {"default": 896, "min": 64, "max": 4096, "step": 16}),
            "height": ("INT", {"default": 1120, "min": 64, "max": 4096, "step": 16}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1,
                             "tooltip": "Random choices use it; auto uses it to pick one of the 3 best tones."}),
        }, "optional": {
            "image": ("IMAGE", {"tooltip": "Photo of the animal, for the automatic choice."}),
            "mask": ("MASK", {"tooltip": "Mask of the animal on that photo."}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("canvas", "color_name")
    FUNCTION = "make"
    CATEGORY = "image"

    def make(self, color, width, height, seed, image=None, mask=None):
        if color == AUTO:
            if image is None:
                color = "pale warm cream"
            else:
                img = image[0]
                m = torch.ones(img.shape[:2]) if mask is None else mask[0]
                if m.shape != img.shape[:2]:
                    m = F.interpolate(m[None, None].float(), size=img.shape[:2], mode="bilinear")[0, 0]
                px = img[m.to(img.device) > 0.5]
                coat = (px.median(dim=0).values if len(px) else img.reshape(-1, 3).median(dim=0).values).tolist()
                color = _auto_pastel(coat, seed)
        elif color in RANDOM:
            names = RANDOM[color]
            color = names[torch.randint(len(names), (1,), generator=torch.Generator().manual_seed(seed)).item()]
        pixel = torch.tensor(_rgb(PASTEL_PALETTE[color]), dtype=torch.float32)
        return (pixel.expand(1, height, width, 3).clone(), color)


class PetFaceMask:
    """Per-pixel strength for the painterly pass (Differential Diffusion + Set Latent Noise Mask):
    1 on fur and body, `face_strength` on the face box, so the eyes and nose stay as painted by FLUX."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "Sets the mask size."}),
            "face_box": ("STRING", {"forceInput": True, "tooltip": 'Text with one or more [x1, y1, x2, y2] in 0-1000 coordinates.'}),
            "face_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                        "tooltip": "Share of the pass's denoise applied to the face: 0.5 with denoise 0.48 repaints the face at ~0.24."}),
            "grow": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                               "tooltip": "Enlarge the face box by this share of its size."}),
            "feather": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                                  "tooltip": "Width of the transition to full strength, as a share of the face size."}),
        }}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "make"
    CATEGORY = "mask"

    def make(self, image, face_box, face_strength, grow, feather):
        h, w = image.shape[1:3]
        mask = torch.ones(1, h, w)
        xs = torch.arange(w, dtype=torch.float32) + 0.5
        ys = torch.arange(h, dtype=torch.float32) + 0.5
        # one box per face, so a group portrait protects every animal's face
        for box in re.findall(r"\[([^\[\]]*)\]", face_box):
            nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", box)[:4]]
            if len(nums) != 4 or nums[2] <= nums[0] or nums[3] <= nums[1]:
                continue
            x0, y0, x1, y1 = nums[0] * w / 1000, nums[1] * h / 1000, nums[2] * w / 1000, nums[3] * h / 1000
            gx, gy = (x1 - x0) * grow / 2, (y1 - y0) * grow / 2
            x0, y0, x1, y1 = x0 - gx, y0 - gy, x1 + gx, y1 + gy
            f = max(1.0, feather * min(x1 - x0, y1 - y0))
            # distance outside the box, 0 inside; full strength once it reaches the feather width
            dx = (x0 - xs).clamp(min=0) + (xs - x1).clamp(min=0)
            dy = (y0 - ys).clamp(min=0) + (ys - y1).clamp(min=0)
            t = (torch.sqrt(dy[:, None] ** 2 + dx[None, :] ** 2) / f).clamp(0, 1)
            t = t * t * (3 - 2 * t)
            mask[0] = torch.minimum(mask[0], face_strength + (1 - face_strength) * t)
        return (mask,)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class PetSplitAnimals:
    """One crop per animal, from Qwen3-VL boxes ("bbox_2d" in 0-1000 coordinates). The output is a
    list, so every node after it runs once per animal and the workflow saves one portrait each.
    With one animal (or none found) the photo passes through unchanged."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "boxes": ("STRING", {"forceInput": True, "tooltip": 'Text with one or more "bbox_2d": [x1, y1, x2, y2].'}),
            "pad": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01,
                              "tooltip": "Margin added around each box, as a share of the box size."}),
            "min_size": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01,
                                   "tooltip": "Skip animals smaller than this share of the largest one (background pets, reflections)."}),
            "max_animals": ("INT", {"default": 6, "min": 1, "max": 20}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("animals",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "split"
    CATEGORY = "image/transform"

    def split(self, image, boxes, pad, min_size, max_animals):
        img = image[:1]
        h, w = img.shape[1:3]
        found = []
        for m in re.finditer(r'"bbox_2d"\s*:\s*\[([^\]]*)\]', boxes):
            nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", m.group(1))[:4]]
            if len(nums) == 4 and nums[2] > nums[0] and nums[3] > nums[1]:
                found.append((nums[0] * w / 1000, nums[1] * h / 1000, nums[2] * w / 1000, nums[3] * h / 1000))
        found.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        kept = []
        for b in found:
            if all(_iou(b, k) < 0.6 for k in kept):  # the model sometimes lists one animal twice
                kept.append(b)
        if kept:
            largest = (kept[0][2] - kept[0][0]) * (kept[0][3] - kept[0][1])
            kept = [b for b in kept if (b[2] - b[0]) * (b[3] - b[1]) >= min_size * largest][:max_animals]
        if len(kept) < 2:
            return ([img],)
        kept.sort(key=lambda b: b[0])  # left to right, so file numbers follow the photo
        crops = []
        for x0, y0, x1, y1 in kept:
            px, py = (x1 - x0) * pad, (y1 - y0) * pad
            cx0, cy0 = max(0, int(x0 - px)), max(0, int(y0 - py))
            cx1, cy1 = min(w, int(x1 + px + 0.5)), min(h, int(y1 + py + 0.5))
            crops.append(img[:, cy0:cy1, cx0:cx1].clone())
        return (crops,)


class PetBustMask:
    """Keep only the head and the upper chest of the animal mask: everything lower than `chest` head
    heights below the chin is removed, with a soft fade (so paws, legs and the body leave the picture)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK",), "head_box": ("STRING", {"forceInput": True}),
                             "chest": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 3.0, "step": 0.05}),
                             "fade": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05})},
                "optional": {"width": ("FLOAT", {"default": 1.7, "min": 0.0, "max": 5.0, "step": 0.05,
                                                 "tooltip": "Max bust width in head widths (0 = no limit)."}),
                             "points": ("STRING", {"forceInput": True,
                                                   "tooltip": "Eyes and nose (Qwen3-VL point_2d) for a reliable chin line."})}}

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "head_box")
    FUNCTION = "apply"
    CATEGORY = "pet_portrait"

    def apply(self, mask, head_box, chest, fade, points=None, width=1.7):
        outs = []
        self.boxes = []
        for m in mask:
            H, W = m.shape
            bb = _bbox(m)
            if bb is None:
                outs.append(m); continue
            y0, y1 = bb[0], bb[1]
            boxes = _boxes(head_box, W, H)
            hb = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if boxes else None
            bh = y1 - y0
            ok = hb is not None and (hb[3] - hb[1]) < 0.7 * bh and hb[1] - y0 < 0.3 * bh
            if ok:
                # a face-only box (ears left out) is stretched up to the top of the animal
                hy0, hy1 = min(hb[1], y0 + 0.02 * bh), hb[3]
                ok = hy1 - hy0 > 0.15 * bh
            if ok:
                hx0, hx1 = hb[0], hb[2]
            else:  # no usable head box: assume the head is the top 40 % of the animal
                hy0, hy1 = y0, y0 + 0.4 * bh
                hx0, hx1 = bb[2], bb[3]
            fp = _face_points(points, W, H) if points and not ok else None
            if fp is not None:
                # eyes and nose are found far more reliably than a head box: the chin lies about
                # 0.9 eye-to-nose distances below the nose, the ears start at the top of the animal
                ex, ey = (fp[0][0] + fp[1][0]) / 2, (fp[0][1] + fp[1][1]) / 2
                nx, ny = fp[2]
                if ny > ey:
                    hy0 = y0
                    hy1 = min(ny + 0.9 * ((nx - ex) ** 2 + (ny - ey) ** 2) ** 0.5, y1)
                    d = abs(fp[1][0] - fp[0][0])
                    cx = (fp[0][0] + fp[1][0]) / 2
                    hx0, hx1 = max(bb[2], cx - 1.3 * d), min(bb[3], cx + 1.3 * d)
            hh = max(hy1 - hy0, 1.0)
            # chest measured against the smaller of head height and head width: a cat photographed from
            # above has a "tall" head on the photo, and the chest would reach the paws
            hh = min(hh, max(hx1 - hx0, 1.0))
            self.boxes.append('"bbox_2d": [%d, %d, %d, %d]' % (hx0 * 1000 / W, hy0 * 1000 / H,
                                                              hx1 * 1000 / W, hy1 * 1000 / H))
            cut = hy1 + chest * hh
            f = max(fade * hh, 1.0)
            ys = torch.arange(H, dtype=m.dtype, device=m.device)[:, None]
            w = ((cut - ys) / f).clamp(0, 1)
            w = w * w * (3 - 2 * w)
            if width > 0 and hx1 - hx0 < 0.95 * (bb[3] - bb[2]):
                # the bust is at most `width` head widths wide, centred on the head: a back or a
                # side of the body next to the head leaves the picture
                hw = hx1 - hx0
                cx = (hx0 + hx1) / 2
                xs = torch.arange(W, dtype=m.dtype, device=m.device)[None, :]
                half = width * hw / 2
                wx = ((half - (xs - cx).abs()) / max(0.15 * hw, 1.0)).clamp(0, 1)
                wx = wx * wx * (3 - 2 * wx)
                # only below the eyes, so ears and the top of the head are never cut
                wx = torch.where(ys < hy0 + 0.5 * hh, torch.ones_like(wx), wx)
                w = w * wx
            outs.append(m * w)
        return (torch.stack(outs), self.boxes[-1] if self.boxes else head_box)


class PetRemoveAccessory:
    """Paint collars, bandanas, harnesses and tags out of the photo: the boxes found by the VLM
    (inside the animal mask) are filled from the surrounding fur, the painterly pass then adds strokes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "mask": ("MASK",), "boxes": ("STRING", {"forceInput": True}),
                             "grow": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.2, "step": 0.01})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "pet_portrait"

    def apply(self, image, mask, boxes, grow):
        outs = []
        for i in range(image.shape[0]):
            img = image[i]
            H, W = img.shape[:2]
            m = mask[min(i, mask.shape[0] - 1)]
            if m.shape != (H, W):
                m = F.interpolate(m[None, None], size=(H, W), mode="bilinear")[0, 0]
            hole = torch.zeros(H, W, dtype=img.dtype, device=img.device)
            g = grow * max(H, W)
            for x1, y1, x2, y2 in _boxes(boxes, W, H):
                if (x2 - x1) * (y2 - y1) > 0.5 * H * W:
                    continue  # a box around the whole animal is not an accessory
                hole[max(0, int(y1 - g)):int(y2 + g), max(0, int(x1 - g)):int(x2 + g)] = 1
            hole = hole * (m > 0.5).to(img.dtype)
            if hole.sum() < 1:
                outs.append(img); continue
            known = (1 - hole) * (m > 0.5).to(img.dtype)  # fill only from the animal, not the backdrop
            x = (img * known[..., None]).permute(2, 0, 1)[None]
            k = known[None, None]
            for _ in range(200):
                if k[0, 0][hole > 0].min() > 0:
                    break
                s = F.avg_pool2d(x, 7, 1, 3)
                c = F.avg_pool2d(k, 7, 1, 3)
                new = (c > 0) & (k == 0)
                x = torch.where(new, s / c.clamp(min=1e-6), x)
                k = torch.where(new, torch.ones_like(k), k)
            fill = F.avg_pool2d(x, 9, 1, 4)[0].permute(1, 2, 0)
            hm = F.avg_pool2d(hole[None, None], 9, 1, 4)[0, 0].clamp(0, 1)[..., None]
            outs.append(img * (1 - hm) + fill * hm)
        return (torch.stack(outs),)


class PetClipToMask:
    """Put the painted animal back onto the clean canvas only inside the animal's outline from the
    photo (slightly grown, so painted wisps of fur survive) and dissolve the bottom of the bust in
    uneven brush-like strokes. Removes anything the painter added around the animal."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"painting": ("IMAGE",), "mask": ("MASK",), "canvas": ("IMAGE",),
                             "grow": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.2, "step": 0.005}),
                             "feather": ("FLOAT", {"default": 0.012, "min": 0.0, "max": 0.1, "step": 0.002}),
                             "bottom_fade": ("FLOAT", {"default": 0.22, "min": 0.0, "max": 0.8, "step": 0.01}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 31 - 1})}}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "apply"
    CATEGORY = "pet_portrait"

    def apply(self, painting, mask, canvas, grow, feather, bottom_fade, seed):
        P = painting[0]
        H, W = P.shape[:2]
        m = mask[:1]
        if m.shape[1:] != (H, W):
            m = F.interpolate(m[None], size=(H, W), mode="bilinear")[0]
        C = canvas[:1].permute(0, 3, 1, 2)
        if C.shape[2:] != (H, W):
            C = F.interpolate(C, size=(H, W), mode="bilinear")
        C = C[0].permute(1, 2, 0)
        a = (m > 0.3).to(P.dtype)[None]
        g = max(1, int(grow * H)) | 1
        a = F.max_pool2d(a, 2 * g + 1, 1, g)
        f = max(1, int(feather * H))
        a = F.avg_pool2d(F.pad(a, (f, f, f, f), mode="replicate"), 2 * f + 1, 1)[0, 0]
        bb = _bbox(m[0])
        if bb is not None and bottom_fade > 0:
            y0, y1 = bb[0], bb[1]
            zone = max(bottom_fade * (y1 - y0), 4.0)
            gen = torch.Generator().manual_seed(seed)
            # uneven end line: smooth random offsets across the width, like the ends of brushstrokes
            n = torch.rand(1, 1, max(4, W // 24), generator=gen)
            n = F.interpolate(n, size=W, mode="linear", align_corners=True)[0, 0]
            fine = F.interpolate(torch.rand(1, 1, max(8, W // 6), generator=gen), size=W, mode="linear",
                                 align_corners=True)[0, 0]
            end = y1 - zone * (0.15 + 0.45 * n + 0.15 * fine)
            ys = torch.arange(H, dtype=P.dtype)[:, None]
            w = ((end[None, :].to(P.dtype) - ys) / (0.6 * zone)).clamp(0, 1)
            w = w * w * (3 - 2 * w)
            a = a * w.to(a.device)
        out = C * (1 - a[..., None]) + P * a[..., None]
        return (out[None], a[None])


class PetLevelHead:
    """Straighten a head tilted towards the shoulder: the photo is rotated around the eyes so the eye
    line becomes (more) horizontal. A plain rotation, nothing is redrawn, every spot stays exact.
    Also returns the eye and nose points moved with the picture."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "mask": ("MASK",), "points": ("STRING", {"forceInput": True}),
                             "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                                                    "tooltip": "Share of the tilt removed (1 = eyes exactly level)."}),
                             "max_angle": ("FLOAT", {"default": 40.0, "min": 0.0, "max": 90.0, "step": 1.0})}}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "points")
    FUNCTION = "apply"
    CATEGORY = "pet_portrait"

    def apply(self, image, mask, points, strength, max_angle):
        H, W = image.shape[1:3]
        fp = _face_points(points, W, H)
        if fp is None:
            return (image, mask, points)
        (lx, ly), (rx, ry), (nx, ny) = fp
        a = math.atan2(ry - ly, rx - lx)
        if abs(math.degrees(a)) > max_angle or abs(math.degrees(a)) < 1.0:
            return (image, mask, points)
        t = a * strength
        cx, cy = (lx + rx) / 2, (ly + ry) / 2
        c, s = math.cos(t), math.sin(t)
        ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32),
                                indexing="ij")
        sx = cx + c * (xs - cx) - s * (ys - cy)
        sy = cy + s * (xs - cx) + c * (ys - cy)
        grid = torch.stack([2 * sx / (W - 1) - 1, 2 * sy / (H - 1) - 1], -1)[None].to(image.device)
        img = F.grid_sample(image.permute(0, 3, 1, 2), grid.expand(image.shape[0], -1, -1, -1),
                            mode="bilinear", padding_mode="border", align_corners=True).permute(0, 2, 3, 1)
        m = mask
        if m.shape[1:] != (H, W):
            m = F.interpolate(m[:, None], size=(H, W), mode="bilinear")[:, 0]
        m = F.grid_sample(m[:, None], grid.expand(m.shape[0], -1, -1, -1), mode="bilinear",
                          padding_mode="zeros", align_corners=True)[:, 0]

        def move(px, py):  # source -> output: rotate back by t
            return (cx + c * (px - cx) + s * (py - cy), cy - s * (px - cx) + c * (py - cy))

        pts = [move(lx, ly), move(rx, ry), move(nx, ny)]
        labels = ["left eye", "right eye", "tip of the nose"]
        text = json.dumps([{"point_2d": [round(x * 1000 / W), round(y * 1000 / H)], "label": lb}
                           for (x, y), lb in zip(pts, labels)])
        print("[PetLevelHead] tilt %.1f deg, rotated by %.1f" % (math.degrees(a), math.degrees(t)))
        return (img, m, text)


class PetPoseHint:
    """Instruction for the painter from the head pose: keep it, or turn the head slightly towards the
    viewer when it is clearly turned away (sideways shift of the nose against the eyes)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"points": ("STRING", {"forceInput": True}),
                             "threshold": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 2.0, "step": 0.01}),
                             "keep_text": ("STRING", {"multiline": True, "default":
                                 "Keep the head pose and angle exactly as in the reference, do not turn the head."}),
                             "turn_text": ("STRING", {"multiline": True, "default":
                                 "Turn the head only slightly, by about 20 degrees, towards the viewer, so the face "
                                 "looks a little more frontal; every marking keeps its place on the face."})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "apply"
    CATEGORY = "pet_portrait"

    def apply(self, points, threshold, keep_text, turn_text):
        fp = _face_points(points, 1000, 1000)
        if fp is None:
            return (keep_text,)
        yaw = _pose(fp)[0]
        print("[PetPoseHint] yaw %.2f" % yaw)
        return (turn_text if abs(yaw) > threshold else keep_text,)


_RAFT = {}


def _gauss(x, s):
    r = max(1, int(3 * s))
    k = torch.exp(-torch.arange(-r, r + 1, dtype=x.dtype, device=x.device) ** 2 / (2 * s * s))
    k = k / k.sum()
    c = x.shape[1]
    x = F.pad(x, (r, r, r, r), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(x, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def _raft_flow(a, b, size=512):
    """Optical flow a -> b (a(p) ~ b(p + flow)) in pixels of a, NCHW images in 0..1; None if unavailable."""
    try:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    except Exception:
        return None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if "m" not in _RAFT:
        _RAFT["m"] = raft_large(weights=Raft_Large_Weights.DEFAULT).eval()
    m = _RAFT["m"].to(dev)
    H, W = a.shape[2:]
    h = size
    w = max(8, int(round(size * W / H / 8)) * 8)
    sm = lambda x: F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False).to(dev) * 2 - 1
    with torch.no_grad():
        f = m(sm(a), sm(b), num_flow_updates=24)[-1].float().cpu()
    m.to("cpu")
    f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
    f[:, 0] *= W / w
    f[:, 1] *= H / h
    return f


def _warp(img, flow):
    H, W = img.shape[2:]
    ys, xs = torch.meshgrid(torch.arange(H, dtype=img.dtype), torch.arange(W, dtype=img.dtype), indexing="ij")
    g = torch.stack([(xs + flow[0, 0]) * 2 / (W - 1) - 1, (ys + flow[0, 1]) * 2 / (H - 1) - 1], -1)[None]
    return F.grid_sample(img, g, padding_mode="border", align_corners=True)


class PetDetailTransfer:
    """Hybrid 'filter': the painting is warped onto the photo by optical flow (RAFT), then the large-scale
    colours, patches, stripes and spots come from the photo and the fine brushwork from the painting.
    Where the flow is unreliable (the painter drew that part elsewhere) the painting is kept as is."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"painting": ("IMAGE",), "photo": ("IMAGE",), "mask": ("MASK",),
                             "detail_size": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.5,
                                                       "tooltip": "Blur radius (px): coarser than this comes from the photo."}),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                             "max_error": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 20.0, "step": 0.5,
                                                     "tooltip": "Forward-backward flow mismatch (px) above which the painting is kept."})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "pet_portrait"

    def apply(self, painting, photo, mask, detail_size, strength, max_error):
        P = painting[:1].permute(0, 3, 1, 2).float().cpu()
        H, W = P.shape[2:]
        Q = photo[:1].permute(0, 3, 1, 2).float().cpu()
        if Q.shape[2:] != (H, W):
            Q = F.interpolate(Q, size=(H, W), mode="bilinear")
        m = mask[:1].float().cpu()
        if m.shape[1:] != (H, W):
            m = F.interpolate(m[None], size=(H, W), mode="bilinear")[0]
        fw = _raft_flow(Q, P)
        if fw is None:
            print("[PetDetailTransfer] torchvision RAFT unavailable, painting kept")
            return (painting,)
        bw = _raft_flow(P, Q)
        fw = _gauss(fw, 3.0)
        warped = _warp(P, fw)
        # forward-backward check: a point taken to the painting and back must land where it started
        err = (fw + _warp(bw, fw)).norm(dim=1, keepdim=True)
        conf = ((2 * max_error - err) / max_error).clamp(0, 1)
        conf = _gauss(conf, 4.0) * m[None] * strength
        hyb = (_gauss(Q, detail_size) + warped - _gauss(warped, detail_size)).clamp(0, 1)
        out = P * (1 - conf) + hyb * conf
        inside = m[0] > 0.5
        print("[PetDetailTransfer] flow %.1f px, reliable %.0f%% of the animal"
              % (fw[0].norm(dim=0)[inside].mean(), 100 * (conf[0, 0][inside] > 0.5).float().mean()))
        return (out.permute(0, 2, 3, 1).to(painting.device, painting.dtype),)


class PetHeadOnBust:
    """Keep only the head from the photo composition and give the painter a classic bust to paint below
    it: a symmetric chest silhouette in the coat colour, from the chin down past the bottom edge.
    Outputs the template, a strength mask for Differential Diffusion (head gentle, body free) and the
    head mask (where the photo must stay exact)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "mask": ("MASK",), "canvas": ("IMAGE",),
                             "points": ("STRING", {"forceInput": True}),
                             "head_strength": ("FLOAT", {"default": 0.83, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "head_half_width": ("FLOAT", {"default": 1.6, "min": 0.5, "max": 4.0, "step": 0.05,
                                                           "tooltip": "Half width of the head in eye distances."}),
                             "chin": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 3.0, "step": 0.05,
                                                "tooltip": "Chin below the nose, in eye-to-nose distances."}),
                             "chest_width": ("FLOAT", {"default": 1.7, "min": 0.5, "max": 4.0, "step": 0.05,
                                                       "tooltip": "Chest width at the bottom edge, in head widths."})}}

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "strength", "head_mask")
    FUNCTION = "apply"
    CATEGORY = "pet_portrait"

    def apply(self, image, mask, canvas, points, head_strength, head_half_width, chin, chest_width):
        img = image[0]
        H, W = img.shape[:2]
        m = mask[0]
        if m.shape != (H, W):
            m = F.interpolate(m[None, None], size=(H, W), mode="bilinear")[0, 0]
        C = canvas[0]
        if C.shape[:2] != (H, W):
            C = F.interpolate(C.permute(2, 0, 1)[None], size=(H, W), mode="bilinear")[0].permute(1, 2, 0)
        fp = _face_points(points, W, H)
        if fp is None:
            print("[PetHeadOnBust] no eyes/nose found, composition kept")
            full = torch.ones(1, H, W)
            return (image, full, mask)
        (lx, ly), (rx, ry), (nx, ny) = fp
        ex, ey = (lx + rx) / 2, (ly + ry) / 2
        d = max(((rx - lx) ** 2 + (ry - ly) ** 2) ** 0.5, 4.0)
        dn = max(((nx - ex) ** 2 + (ny - ey) ** 2) ** 0.5, 2.0)
        yc = ny + chin * dn  # chin line
        cx = ex + 0.3 * (nx - ex)
        hw = head_half_width * d
        ys = torch.arange(H, dtype=img.dtype)[:, None].expand(H, W)
        xs = torch.arange(W, dtype=img.dtype)[None, :].expand(H, W)
        band = max(0.25 * dn, 3.0)
        # head = animal mask above the chin and within the head width, soft edges
        above = ((yc - ys) / band).clamp(0, 1)
        inside = ((hw - (xs - cx).abs()) / max(0.25 * d, 2.0)).clamp(0, 1)
        inside = torch.where(ys < ey, torch.ones_like(inside), inside)  # never trim the ears
        head = (m > 0.3).to(img.dtype) * above * inside
        head = F.avg_pool2d(head[None, None], 5, 1, 2)[0, 0]
        # neck/chest silhouette: from a neck a bit narrower than the head down to the bottom edge
        top = yc - 0.6 * dn
        t = ((ys - top) / max(H - top, 1.0)).clamp(0, 1)
        half = hw * (0.75 + (chest_width - 0.75) * t ** 0.7)
        sil = ((half - (xs - cx).abs()) / max(0.3 * d, 2.0)).clamp(0, 1) * ((ys - top) / band).clamp(0, 1)
        # coat colour from the lower part of the head (cheeks, chin) on the photo
        sample = (m > 0.5) & (ys > ey) & (ys < yc) & ((xs - cx).abs() < hw)
        col = img[sample].median(0).values if sample.any() else img[m > 0.5].median(0).values
        body = C * (1 - sil[..., None]) + col * sil[..., None]
        out = body * (1 - head[..., None]) + img * head[..., None]
        strength = head_strength * head + 1.0 * (1 - head)
        strength = F.avg_pool2d(strength[None, None], 9, 1, 4)[0, 0]
        print("[PetHeadOnBust] eyes %.0f px apart, chin at %.0f of %d" % (d, yc, H))
        return (out[None], strength[None], head[None])


class PetSingleAnimal:
    """True when the photo holds a single animal (the list from Pet Split Animals has one item)."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"animals": ("IMAGE",)}}

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "apply"
    CATEGORY = "pet_portrait"

    def apply(self, animals):
        return (len(animals) == 1,)


NODE_CLASS_MAPPINGS = {"PetPortraitLayout": PetPortraitLayout, "PetHeadPortraitLayout": PetHeadPortraitLayout,
                       "PetRestoreColors": PetRestoreColors, "PetFlattenBackground": PetFlattenBackground,
                       "PetPastelBackground": PetPastelBackground, "PetPastelPalette": PetPastelPalette,
                       "PetPortraitLayoutPadded": PetPortraitLayoutPadded, "PetColorPalette": PetColorPalette,
                       "PetSplitAnimals": PetSplitAnimals, "PetFaceMask": PetFaceMask, "PetGroupLayout": PetGroupLayout,
                       "PetMatchColors": PetMatchColors, "PetCropToBox": PetCropToBox,
                       "PetTransferSpots": PetTransferSpots, "PetFaceFromPhoto": PetFaceFromPhoto,
                       "PetBustMask": PetBustMask, "PetRemoveAccessory": PetRemoveAccessory,
                       "PetClipToMask": PetClipToMask, "PetSingleAnimal": PetSingleAnimal,
                       "PetLevelHead": PetLevelHead, "PetPoseHint": PetPoseHint,
                       "PetDetailTransfer": PetDetailTransfer, "PetHeadOnBust": PetHeadOnBust}
NODE_DISPLAY_NAME_MAPPINGS = {"PetPortraitLayout": "Pet Portrait Layout", "PetHeadPortraitLayout": "Pet Head Portrait Layout",
                              "PetRestoreColors": "Pet Restore Colors", "PetFlattenBackground": "Pet Flatten Background",
                              "PetPastelBackground": "Pet Pastel Background",
                              "PetPastelPalette": "Pet Pastel Palette (experimental)",
                              "PetPortraitLayoutPadded": "Pet Portrait Layout (padded, experimental)",
                              "PetColorPalette": "Pet Color Palette (experimental)",
                              "PetSplitAnimals": "Pet Split Animals (experimental)",
                              "PetFaceMask": "Pet Face Mask (experimental)",
                              "PetGroupLayout": "Pet Group Layout (experimental)",
                              "PetMatchColors": "Pet Match Colors (experimental)",
                              "PetCropToBox": "Pet Crop To Box (experimental)",
                              "PetTransferSpots": "Pet Transfer Spots (experimental)",
                              "PetFaceFromPhoto": "Pet Face From Photo (experimental)",
                              "PetBustMask": "Pet Bust Mask (experimental)",
                              "PetRemoveAccessory": "Pet Remove Accessory (experimental)",
                              "PetClipToMask": "Pet Clip To Mask (experimental)",
                              "PetSingleAnimal": "Pet Single Animal (experimental)",
                              "PetLevelHead": "Pet Level Head (experimental)",
                              "PetPoseHint": "Pet Pose Hint (experimental)",
                              "PetDetailTransfer": "Pet Detail Transfer (experimental)",
                              "PetHeadOnBust": "Pet Head On Bust (experimental)"}
