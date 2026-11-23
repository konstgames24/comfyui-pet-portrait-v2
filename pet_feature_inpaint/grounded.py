"""Pet Grounded Layout: every animal stands on the bottom edge, never floats.

Replaces PetGroupLayout (pet_portrait_layout, t6 - not changed) in the feat2 workflow. Why a new node:
  * PetGroupLayout fell back to its old layout when head_height was 0. On RunComfy the frontend loaded
    the UI JSON with one widget value missing after "seed" (control_after_generate), so head_height
    became 0 and every portrait used the old layout.
  * The brushy fade ran out of alpha above the bottom edge in some columns, up to ~5 % of the height.
    It was anchored to the lowest mask pixel. When that pixel was a faint shadow or a stray bit of fur,
    the bust also ended above the edge. A group showed it most, because each animal has its own mask.

Rules, applied to each animal on its own (in a group too):
  1. scale from the head: its height = head_height x canvas height (Qwen3-VL head box). If there is
     no usable box, the whole solid animal is fitted between top_space and the edge instead, so a
     guessed head size never blows the animal up;
  2. the "solid bottom" of the bust is the lowest row that is still at least 80 % as wide as the
     lower bust and painted at least 45 % as strongly as the chest (the colour difference from the
     painting's own background). Stray pixels, thin tails, shadows, a rounded end and a fade-out the
     painter added below it do not count;
  3. that row is placed bottom_overflow below the bottom edge, so solid fur always crosses the edge.
     If the head would then sit higher than top_space, the head is put at top_space instead and the
     bust runs further past the edge. If the bust is too short to reach from top_space to the edge,
     the animal is enlarged (up to max_grow x) and then moved down: more empty space above it,
     never below;
  4. the brushy fade is measured in canvas space. Every stroke ends below the edge, so the alpha
     at the edge stays between ~0.55 and ~0.95 across the bust;
  5. width limit (max_width) and group packing only shrink the animal, and after any shrink rules
     2-4 are applied again.
No widget named seed (so no hidden control widget) and no setting that switches the rules off."""
import re

import torch
import torch.nn.functional as F

GROUP = "group (all on one image)"
SEPARATE = "separate portraits"
# a row counts as solid bust if it is at least this share of the lower bust width: a rounded or tapered
# painted bottom (FLUX often paints one) goes past the edge instead of leaving its corners in the air
SOLID = 0.8


def _fit_mask(mask, size):
    m = mask[:1].float()
    if tuple(m.shape[1:]) != tuple(size):
        m = F.interpolate(m[None], size=size, mode="bilinear")[0]
    return m[0]


def _boxes(text, w, h):
    out = []
    for g in re.findall(r"\[([^\[\]]*)\]", text or ""):
        nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", g)[:4]]
        if len(nums) == 4 and nums[2] > nums[0] and nums[3] > nums[1]:
            out.append((nums[0] * w / 1000, nums[1] * h / 1000, nums[2] * w / 1000, nums[3] * h / 1000))
    return out


def _soften(m, r):
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


def _spans(m, y0, y1):
    """Per row: first and last mask column (or -1, -1) and the span width."""
    b = m[y0:y1] > 0.5
    idx = torch.arange(b.shape[1], device=b.device)[None].expand_as(b)
    left = torch.where(b, idx, torch.full_like(idx, b.shape[1])).min(1).values
    right = torch.where(b, idx, torch.full_like(idx, -1)).max(1).values
    width = (right - left + 1).clamp(min=0).float()
    return left, right, width


def close_lower_bust(m, y0, y1, share=0.4):
    """Fill gaps across each row of the lowest `share` of the animal (a light chest the background
    remover missed would otherwise leave holes open onto the bottom edge)."""
    ya = int(y1 - share * (y1 - y0))
    left, right, width = _spans(m, ya, y1)
    xs = torch.arange(m.shape[1], device=m.device)[None]
    fill = ((xs >= left[:, None]) & (xs <= right[:, None]) & (width[:, None] > 0)).to(m)
    m = m.clone()
    m[ya:y1] = torch.maximum(m[ya:y1], fill)
    return m


PAINT = 0.45  # ... and its fur differs from the painting's background at least this much (vs the chest)


def _paint_strength(img, m, y0, y1):
    """Per row of the animal: median colour distance of its pixels from the painting's own background
    (low where the painter faded the bust out into the background). None if no background is visible."""
    out = m < 0.1
    if out.sum() < 500:
        return None
    bg = img[out].median(0).values
    d = (img[y0:y1] - bg).abs().sum(-1)
    d = torch.where(m[y0:y1] > 0.5, d, torch.full_like(d, float("nan")))
    return torch.nanmedian(d, dim=1).values


def solid_bottom(m, y0, y1, img=None):
    """Lowest row (exclusive) of the bust that is still SOLID x as wide as the lower bust and, with
    img, painted at least PAINT x as strongly as the chest (not faded out into the background)."""
    rows = _spans(m, y0, y1)[2]
    lower = rows[len(rows) // 2:]
    lower = lower[lower > 0]
    if len(lower) == 0:
        return y1
    ref = torch.quantile(lower, 0.75).item()
    ok = rows >= SOLID * ref
    ps = _paint_strength(img, m, y0, y1) if img is not None else None
    if ps is not None:
        n = len(ps)
        chest = ps[int(0.5 * n):int(0.85 * n)]
        chest = chest[~torch.isnan(chest)]
        if len(chest):
            ok = ok & (torch.nan_to_num(ps, nan=0.0) >= PAINT * chest.median().item())
    idx = torch.nonzero(ok, as_tuple=True)[0]
    ysb = y0 + idx.max().item() + 1 if len(idx) else y1
    # never drop more than 45 % of the animal as "not solid": then the detection, not the bust, is wrong
    return max(ysb, int(y0 + 0.55 * (y1 - y0)))


def grounded_fade(shape, y0, edge, fade, variant):
    """Alpha over source rows: 1 above the fade, a ragged brushy fade towards the canvas bottom edge
    (source row `edge`). Every column's stroke ends below the edge: alpha there is 0.57-0.94."""
    h, w = shape
    if fade <= 0:
        return torch.ones(h, w)
    g = torch.Generator().manual_seed(int(variant))

    def noise(knots):
        v = torch.rand(1, 1, knots, generator=g)
        return F.interpolate(v, size=w, mode="linear", align_corners=True)[0, 0]

    j = 0.65 * noise(9) + 0.35 * noise(70)  # 0..1
    length = max(2.0, fade * (edge - y0))
    end = edge + length * (0.55 + 0.3 * j)
    rows = torch.arange(h, dtype=torch.float32)[:, None]
    t = ((rows - (end[None, :] - length)) / length).clamp(0, 1)
    return 1 - t * t * (3 - 2 * t)


def _head(m, bb, text, w, h):
    y0, y1, x0, x1 = bb
    bh = y1 - y0
    boxes = _boxes(text, w, h) if text else []
    box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if boxes else None
    # a "head" box covering most of the bust is really the whole animal; one far below the top is wrong
    if box and box[3] - box[1] <= 0.7 * bh and box[1] - y0 <= 0.3 * bh and box[3] - box[1] >= 0.12 * bh:
        return box[0], min(box[1], y0 + 0.02 * bh), box[2], box[3], True
    # no usable box: only the head's horizontal centre is estimated (top third of the animal);
    # the scale then comes from the whole animal, not from a guessed head height
    cols = torch.nonzero((m[y0:int(y0 + bh / 3) + 1] > 0.5).any(0), as_tuple=True)[0]
    hx0, hx1 = (cols.min().item(), cols.max().item() + 1) if len(cols) else (x0, x1)
    return hx0, y0, hx1, y0 + 0.5 * bh, False


def plan_animal(img, m, head_text, ch, top_space, head_height, overflow, max_grow):
    """Everything about one animal that does not depend on its neighbours."""
    ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
    if len(ys) == 0:
        return None
    bb = (ys.min().item(), ys.max().item() + 1, xs.min().item(), xs.max().item() + 1)
    m = close_lower_bust(m, bb[0], bb[1])
    hx0, hy0, hx1, hy1, trusted = _head(m, bb, head_text, img.shape[1], img.shape[0])
    ysb = solid_bottom(m, bb[0], bb[1], img)
    top, line = top_space * ch, ch * (1 + overflow)
    need = (line - top) / max(1.0, ysb - hy0)  # the solid animal spans exactly top_space .. the edge
    if trusted:
        s = head_height * ch / max(1.0, hy1 - hy0)
        # bust too short to reach from top_space to below the edge: grow it (a little), then move it down
        if s < need:
            s = min(need, s * max_grow)
    else:
        s = need
    return {"img": img, "m": m, "y0": bb[0], "y1": bb[1], "ysb": ysb, "hy0": hy0,
            "hx": (hx0 + hx1) / 2, "s": s, "top": top, "line": line, "ch": ch}


def fit(p):
    """Vertical placement and the visible part for the current scale p["s"]."""
    s = p["s"]
    p["oy"] = max(p["top"] - p["hy0"] * s, p["line"] - p["ysb"] * s)
    p["edge"] = (p["ch"] - p["oy"]) / s  # source row at the canvas bottom edge
    ye = int(min(p["y1"], p["edge"] + 1))
    cols = torch.nonzero((p["m"][p["y0"]:max(ye, p["y0"] + 1)] > 0.5).any(0), as_tuple=True)[0]
    p["vx0"], p["vx1"] = cols.min().item(), cols.max().item() + 1
    return p


def limit_width(p, max_px):
    fit(p)
    for _ in range(3):
        wv = (p["vx1"] - p["vx0"]) * p["s"]
        if wv <= max_px:
            break
        p["s"] *= max_px / wv
        fit(p)
    return p


def paste_animal(cv, alpha, p, ox, fade, variant, softness):
    m = p["m"] * grounded_fade(p["m"].shape, p["y0"], p["edge"], fade, variant).to(p["m"])
    ox, oy = round(ox), round(p["oy"])
    p["ox"], p["fade_mask"] = ox, m
    cv = _paste(cv, p["img"], m, p["s"], ox, oy, softness)
    a = _paste(torch.zeros(cv.shape[0], cv.shape[1], 1).to(cv), torch.ones_like(p["img"][..., :1]), m,
               p["s"], ox, oy, softness)
    return cv, torch.maximum(alpha, a)


def layout(images, masks, canvases, head_texts, mode, top_space, head_height, overflow, fade, max_width,
           overlap, max_aspect, softness, variant, max_grow=1.35):
    """images: list of HxWx3, masks: list of HxW, canvases: list of 1xHxWx3 (one per animal or one).
    Returns (list of 1xHxWx3 images, list of 1xHxW alphas, list of canvases, list of plans)."""
    n = len(images)
    canvases = [canvases[min(i, len(canvases) - 1)] for i in range(n)]
    ch, cw0 = canvases[0].shape[1:3]
    parts = []
    for i in range(n):
        img = images[i]
        m = _fit_mask(masks[min(i, len(masks) - 1)], img.shape[:2]).to(img)
        text = head_texts[min(i, len(head_texts) - 1)] if head_texts else ""
        p = plan_animal(img, m, text, ch, top_space, head_height, overflow, max_grow)
        if p is not None:
            p["i"] = i
            parts.append(limit_width(p, max_width * cw0))
    if not parts:
        return canvases, [torch.zeros(1, ch, cw0) for _ in canvases], canvases, []

    if mode == SEPARATE or len(parts) == 1:
        imgs, alphas, cvs = [], [], []
        for p in parts:
            cv = canvases[p["i"]][0].to(p["img"]).clone()
            # centre between the head and the visible bust, so the head stays near the middle
            cx = (p["hx"] + (p["vx0"] + p["vx1"]) / 2) / 2
            cv, a = paste_animal(cv, torch.zeros(ch, cw0, 1).to(cv), p, cw0 / 2 - cx * p["s"], fade,
                                 variant + p["i"], softness)
            imgs.append(cv[None]); alphas.append(a[None, ..., 0]); cvs.append(canvases[p["i"]])
        return imgs, alphas, cvs, parts

    widths = [(p["vx1"] - p["vx0"]) * p["s"] for p in parts]
    total = sum(widths) - overlap * sum(widths[1:])
    cw = min(max(cw0, total / max_width), ch * max_aspect)
    cw = max(64, int(round(cw / 64)) * 64)
    k = min(1.0, max_width * cw / total)
    if k < 1:  # shrinking moves every animal back onto the edge (fit), one by one
        for p in parts:
            p["s"] *= k
            fit(p)
        widths = [(p["vx1"] - p["vx0"]) * p["s"] for p in parts]
        total = sum(widths) - overlap * sum(widths[1:])
    pixel = canvases[0][0, 0, 0].to(parts[0]["img"])
    cv = pixel.expand(ch, cw, 3).clone()
    alpha = torch.zeros(ch, cw, 1).to(cv)
    x = (cw - total) / 2
    placed = []
    for p, wd in zip(parts, widths):
        placed.append((p, x - p["vx0"] * p["s"]))
        x += wd * (1 - overlap)
    mid = (len(placed) - 1) / 2
    for j in sorted(range(len(placed)), key=lambda i: -abs(i - mid)):  # middle ones in front
        p, ox = placed[j]
        cv, alpha = paste_animal(cv, alpha, p, ox, fade, variant + p["i"], softness)
    return [cv[None]], [alpha[None, ..., 0]], [pixel.expand(1, ch, cw, 3).clone()], parts


class PetGroundedLayout:
    """Layout of one or several painted animals in which every bust starts at the bottom edge.
    Each animal is placed on its own; a group only adds the side-by-side packing."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "canvas": ("IMAGE", {"tooltip": "Background; the group canvas takes the first one's colour and height."}),
            "mode": ([GROUP, SEPARATE],),
            "top_space": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.8, "step": 0.01,
                                    "tooltip": "Least empty space above the head, as a share of the height."}),
            "head_height": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 0.8, "step": 0.01,
                                      "tooltip": "Head height (ears to chin) as a share of the canvas height."}),
            "bottom_overflow": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.3, "step": 0.01,
                                          "tooltip": "How far the solid bust runs past the bottom edge."}),
            "bottom_fade": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.8, "step": 0.01,
                                      "tooltip": "Lower share of the visible bust that fades; it never fades out above the edge."}),
            "max_width": ("FLOAT", {"default": 0.86, "min": 0.2, "max": 1.0, "step": 0.01}),
            "overlap": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.6, "step": 0.01,
                                  "tooltip": "How much neighbours overlap, as a share of their width (group only)."}),
            "max_aspect": ("FLOAT", {"default": 1.6, "min": 0.5, "max": 3.0, "step": 0.05,
                                     "tooltip": "Widest group canvas, as width / height."}),
            "edge_softness": ("INT", {"default": 4, "min": 0, "max": 32}),
            "fade_variant": ("INT", {"default": 0, "min": 0, "max": 9999, "tooltip": "Shape of the brushy fade."}),
        }, "optional": {
            "head_box": ("STRING", {"forceInput": True, "tooltip": "Qwen3-VL head box of each painted animal."}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "mask", "canvas")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "run"
    CATEGORY = "pet_portrait/experimental"

    def run(self, image, mask, canvas, mode, top_space, head_height, bottom_overflow, bottom_fade, max_width,
            overlap, max_aspect, edge_softness, fade_variant, head_box=None):
        imgs, alphas, cvs, parts = layout(
            [x[0] for x in image], list(mask), list(canvas), head_box, mode[0], top_space[0], head_height[0],
            bottom_overflow[0], bottom_fade[0], max_width[0], overlap[0], max_aspect[0], edge_softness[0],
            fade_variant[0])
        for p in parts:
            print("[PetGroundedLayout] animal %d: scale %.3f, solid bottom row %d of %d -> %.0f px below the edge"
                  % (p["i"], p["s"], p["ysb"], p["y1"], p["oy"] + p["ysb"] * p["s"] - p["ch"]))
        return (imgs, alphas, cvs)
