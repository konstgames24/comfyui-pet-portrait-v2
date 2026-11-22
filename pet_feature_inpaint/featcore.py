"""EXPERIMENT (on top of t6): keep the pet's likeness by comparing the painted portrait with the photo
through high-level features and re-painting only the features the painting lost.

1. Both head crops are aligned: the photo is warped into the geometry of the painting.
   Alignment methods ("trackers"): SIFT, DISK+LightGlue, LoFTR keypoints; RAFT dense optical flow;
   semantic patch matching with ViT features (DINOv2, CLIP), optionally refined by RAFT or by a dense
   semantic search.
2. Where the aligned photo and the painting disagree both semantically (DINOv2 / CLIP patch features)
   and in appearance (local colour / tone), a feature of the photo is missing in the painting (or the
   painter invented one): this becomes the inpaint mask.
3. The aligned photo is pasted under the mask with its colours adapted to the painting; a masked SDXL
   pass then repaints it in the brush style (in the workflow), and the head crop goes back in place.

Pure torch + kornia + transformers; no ComfyUI imports, so it can be tested on its own.
Coordinates of the crops are normalised to [-1, 1] (align_corners=True), so every transform works
at any resolution."""
import math
import re

import torch
import torch.nn.functional as F

try:
    from . import transforms as tf
except ImportError:  # imported as a top-level module by the lab scripts
    import transforms as tf

DEV = "cuda" if torch.cuda.is_available() else "cpu"
_M = {}
FACE_PAD = 0.35  # face box (eyes, nose, mouth) -> square crop of the whole head
METHODS = ["dinov2+raft", "dinov2", "dinov2+dense", "clip", "clip+raft", "raft",
           "sift", "loftr", "disk+lightglue", "boxes"]
# geometric model fitted to the matches; chosen on feature_lab/tf_bench (README there): homography by
# RANSAC + smoothing thin-plate spline gave the smallest landmark error; "legacy" = the old RANSAC affine
DEFAULT_TRANSFORM = "homography/ransac+tps(lam=0.1)"
TRANSFORMS = [DEFAULT_TRANSFORM, "legacy", "affine/ransac+tps(lam=0.1)", "homography/ransac+tps(lam=1)",
              "homography/ransac+ffd(grid=6,lam=0.1)", "homography/ransac+ffd(grid=8,lam=0.3)"] + \
    ["%s/%s" % (g, e) for g in ("rigid", "similarity", "affine", "homography", "poly2", "poly3")
     for e in ("ransac", "lmeds", "huber", "tukey", "ransac+tukey")]


# ------------------------------------------------------------------ boxes, masks, crops
def parse_box(text, w, h):
    """First Qwen3-VL box ("bbox_2d": [x1, y1, x2, y2], 0-1000) in pixels, or None."""
    if not text:
        return None
    for m in re.finditer(r"\[([^\[\]]*)\]", text):
        nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", m.group(1))]
        if len(nums) >= 4:
            x0, y0, x1, y1 = nums[0] * w / 1000, nums[1] * h / 1000, nums[2] * w / 1000, nums[3] * h / 1000
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            if x1 - x0 > 4 and y1 - y0 > 4:
                return x0, y0, x1, y1
    return None


def parse_boxes(text, w, h):
    """One box per animal (pixels). Qwen3-VL sometimes nests parts in an entry ("bbox_2d" of the body
    plus "face": {"eyes": [...], "nose": [...], "mouth": [...]}): then the union of the parts is the face."""
    out = []
    for seg in re.split(r'"bbox_2d"', text or "")[1:]:
        boxes = [parse_box("[%s]" % m.group(1), w, h) for m in re.finditer(r"\[([^\[\]]*)\]", seg)]
        boxes = [b for b in boxes if b]
        if not boxes:
            continue
        body, parts = boxes[0], boxes[1:]
        b = body
        if parts:
            b = (min(p[0] for p in parts), min(p[1] for p in parts), max(p[2] for p in parts),
                 max(p[3] for p in parts))
            if b[2] - b[0] < 0.3 * (body[2] - body[0]):  # one eye only: take the top of the body box
                bw = body[2] - body[0]
                b = (body[0], body[1], body[2], min(body[3], body[1] + 0.8 * bw))
        if all(_iou(b, o) < 0.7 for o in out):
            out.append(b)
    return out


def parse_landmarks(text, w, h):
    """Qwen3-VL points ("point_2d": [x, y] in 0-1000, "label": "animal 1 left eye" ...) -> one dict per
    animal {eye_a, eye_b, nose} in pixels, eyes sorted by x. Animals are told apart by the number in
    the label; without numbers the points are taken in threes as they come."""
    pts = []
    for obj in re.findall(r"\{[^{}]*\}", text or ""):  # one JSON object per point, keys in any order
        pm = re.search(r'"point_2d"\s*:\s*\[([^\]]*)\]', obj)
        if not pm:
            continue
        lm = re.search(r'"label"\s*:\s*"([^"]*)"', obj)
        v = [float(t) for t in re.findall(r"-?\d+(?:\.\d+)?", pm.group(1))]
        if len(v) >= 2:
            pts.append((v[0] * w / 1000, v[1] * h / 1000, (lm.group(1) if lm else "").lower()))
    groups = {}
    for i, (x, y, lab) in enumerate(pts):
        num = re.search(r"\d+", lab)
        key = int(num.group()) if num else i // 3
        groups.setdefault(key, []).append((x, y, lab))
    out = []
    for key in sorted(groups):
        g = groups[key]
        nose = [p for p in g if "nose" in p[2]]
        eyes = [p for p in g if "eye" in p[2]]
        if not nose and not eyes and len(g) == 3:  # unlabelled: the two upper points are the eyes
            s = sorted(g, key=lambda p: p[1])
            eyes, nose = s[:2], s[2:]
        if nose and len(eyes) >= 2:
            e = sorted(eyes[:2], key=lambda p: p[0])
            out.append({"eye_a": e[0][:2], "eye_b": e[1][:2], "nose": nose[0][:2]})
    return out


def landmark_box(lm, k=2.2):
    """Face box (pixels) from the eyes and the nose: centred between the eyes' midpoint and the nose,
    side k x the larger of the eye distance and the eyes-to-nose distance (works for a turned head too).
    region_of(box, FACE_PAD) then frames the whole head."""
    ea, eb, n = [torch.tensor(lm[k_], dtype=torch.float32) for k_ in ("eye_a", "eye_b", "nose")]
    mid = (ea + eb) / 2
    c = (mid + n) / 2
    s = k * max(float((eb - ea).norm()), float((n - mid).norm()), 1.0)
    return float(c[0] - s / 2), float(c[1] - s / 2), float(c[0] + s / 2), float(c[1] + s / 2)


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


@torch.no_grad()
def own_mask(c, region, box, others, fg, sem_res=644):
    """Mask (1, 1, s, s) of the animal whose face box is `box` inside the crop `c` of `region`, the
    neighbours (their face boxes in `others`) cut away. DINOv2 patches are clustered from the face
    boxes as seeds (3 k-means steps), then the own label is grown from the own face only, so a
    similar-looking neighbour that does not touch it stays out."""
    if not others:
        return torch.ones_like(fg)
    f = patch_feats(c, "dinov2", sem_res)
    C, n = f.shape[1], f.shape[-1]
    X = f[0].reshape(C, -1).T
    g = ident_grid(n, "cpu")[0]
    cx, cy, side = region

    def inside(b):
        x0, y0, x1, y1 = (b[0] - cx) / (side / 2), (b[1] - cy) / (side / 2), (b[2] - cx) / (side / 2), (b[3] - cy) / (side / 2)
        return (g[..., 0] >= x0) & (g[..., 0] <= x1) & (g[..., 1] >= y0) & (g[..., 1] <= y1)

    fgn = (F.interpolate(fg, size=(n, n), mode="area")[0, 0] > 0.5).reshape(-1)
    seeds = [inside(b).reshape(-1) for b in [box] + list(others)]
    # a patch inside several face boxes is nobody's seed
    cnt = torch.stack(seeds).sum(0)
    seeds = [s & (cnt == 1) & fgn for s in seeds]
    protos = [F.normalize(X[s].mean(0), dim=0) if s.any() else None for s in seeds]
    if protos[0] is None:
        return torch.ones_like(fg)
    protos = [p for p in protos if p is not None]  # the own face is first
    for _ in range(3):
        lab = torch.stack([X @ p for p in protos], 1).argmax(1)
        for j in range(len(protos)):
            sel = (lab == j) & fgn
            if sel.any():
                protos[j] = F.normalize(X[sel].mean(0), dim=0)
    own = ((lab == 0) & fgn).reshape(1, 1, n, n).float()
    own = erode(dilate(own, 2), 2) * fgn.reshape(1, 1, n, n).float()  # close small holes (white muzzle)
    grow = (seeds[0].reshape(1, 1, n, n).float() * own)
    for _ in range(2 * n):  # geodesic growth from the own face inside the own label
        nxt = dilate(grow, 1) * own
        if torch.equal(nxt, grow):
            break
        grow = nxt
    m = F.interpolate(grow, size=fg.shape[2:], mode="bilinear", align_corners=False)
    return (blur(m, fg.shape[-1] / 128) > 0.5).float() * fg


@torch.no_grad()
def head_embed(img, box, others=()):
    reg = region_of(box, FACE_PAD)
    x = crop(img, reg, 448)
    if others:  # the neighbour must not make two heads look alike
        fg = fg_mask(x)
        m = own_mask(x, reg, box, [o for o in others], fg)
        b = torch.cat([x[..., 0, :], x[..., -1, :]], -1).median(-1).values[..., None, None]
        x = x * m + b * (1 - m)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    with _on_gpu("dinov2") as m:
        return F.normalize(m(pixel_values=_norm(x, _IMNET).to(DEV)).pooler_output.float().cpu(), dim=1)[0]


def pair_heads(P, Q, pboxes, qboxes):
    """Which head on the photo is which head on the painting: the assignment with the most similar
    DINOv2 embeddings (several animals; separate portraits get the best photo head)."""
    import itertools
    pboxes = pboxes or [fallback_box(P)]
    qboxes = qboxes or [fallback_box(Q)]
    if len(pboxes) == 1 and len(qboxes) == 1:
        return [(pboxes[0], qboxes[0], None)]
    ep = torch.stack([head_embed(P, b, [o for o in pboxes if o is not b]) for b in pboxes])
    eq = torch.stack([head_embed(Q, b, [o for o in qboxes if o is not b]) for b in qboxes])
    S = ep @ eq.T
    n = min(len(pboxes), len(qboxes))
    if len(pboxes) <= len(qboxes):
        best = max(itertools.permutations(range(len(qboxes)), n), key=lambda p: sum(S[i, j] for i, j in enumerate(p)))
        pairs = [(i, j) for i, j in enumerate(best)]
    else:
        best = max(itertools.permutations(range(len(pboxes)), n), key=lambda p: sum(S[i, j] for j, i in enumerate(p)))
        pairs = [(i, j) for j, i in enumerate(best)]
    return [(pboxes[i], qboxes[j], round(float(S[i, j]), 3)) for i, j in pairs]


def heads_of(text, img):
    """Heads from a Qwen3-VL answer: eye + nose points (FIND_LANDMARKS, preferred: the face box is
    framed from them and they anchor the alignment) or face boxes (FIND_FACE, older workflows).
    -> list of (box, landmarks or None), pixels."""
    w, h = img.shape[3], img.shape[2]
    if "point_2d" in (text or ""):
        lms = parse_landmarks(text, w, h)
        if lms:
            return [(landmark_box(lm), lm) for lm in lms]
    return [(b, None) for b in parse_boxes(text, w, h)]


@torch.no_grad()
def analyse_all(P, Q, painting_text, photo_text, **kw):
    """analyse() for every pair of heads (Qwen landmarks or face boxes, widened to the whole head);
    batched outputs for one SDXL pass."""
    kw.setdefault("pad", FACE_PAD)
    ph, qh = heads_of(painting_text, P), heads_of(photo_text, Q)
    plm = {tuple(b): lm for b, lm in ph}
    qlm = {tuple(b): lm for b, lm in qh}
    pairs = pair_heads(P, Q, [b for b, _ in ph], [b for b, _ in qh])
    res = []
    pall = [p for p, _, _ in pairs] + [b for b, _ in ph if all(b != p for p, _, _ in pairs)]
    qall = [q for _, q, _ in pairs] + [b for b, _ in qh if all(b != q for _, q, _ in pairs)]
    for pb, qb, sim in pairs:
        r = analyse(P, Q, pb, qb, painting_others=[b for b in pall if b != pb],
                    photo_others=[b for b in qall if b != qb], painting_lm=plm.get(tuple(pb)),
                    photo_lm=qlm.get(tuple(qb)), **kw)
        r["report"]["pair_sim"] = sim
        res.append(r)
    debug = torch.cat([r["debug"] for r in res], 2)
    return dict(pasted=torch.cat([r["pasted"] for r in res]), inpaint_mask=torch.cat([r["inpaint_mask"] for r in res]),
                painting_crop=torch.cat([r["painting_crop"] for r in res]), debug=debug,
                regions=[r["region"] for r in res], reports=[r["report"] for r in res])


@torch.no_grad()
def likeness(P, Q, painting_text, photo_text):
    """Mean over head pairs: DINOv2 CLS / CLIP similarity and aligned DINOv2 patch similarity."""
    pairs = pair_heads(P, Q, parse_boxes(painting_text, P.shape[3], P.shape[2]),
                       parse_boxes(photo_text, Q.shape[3], Q.shape[2]))
    acc = {"dino": 0.0, "clip": 0.0, "aligned": 0.0}
    for pb, qb, _ in pairs:
        s = global_sim(crop(P, region_of(pb, FACE_PAD), 448), crop(Q, region_of(qb, FACE_PAD), 448))
        s["aligned"] = analyse(P, Q, pb, qb, method="dinov2+raft", out_size=256, pad=FACE_PAD)["report"]["aligned_sim"]
        for k in acc:
            acc[k] += s[k] / len(pairs)
    return {k: round(v, 4) for k, v in acc.items()}, len(pairs)


def fg_mask(img, thr=0.09):
    """Animal mask of a cut-out / flat-background image (NCHW): differs from the border colour."""
    b = torch.cat([img[..., 0, :], img[..., -1, :], img[..., :, 0], img[..., :, -1]], -1)
    bg = b.median(-1).values[..., None, None]
    d = (img - bg).norm(dim=1, keepdim=True)
    m = (d > thr).float()
    return blur(dilate(m, 2), 1.0)


def fallback_box(img):
    m = fg_mask(img)[0, 0] > 0.5
    ys, xs = torch.nonzero(m, as_tuple=True)
    if len(xs) < 10:
        H, W = img.shape[2:]
        return 0, 0, W, H
    x0, x1 = xs.min().item(), xs.max().item()
    y0 = ys.min().item()
    return x0, y0, x1, min(img.shape[2], y0 + (x1 - x0))  # top square of the animal ~ the head


def region_of(box, pad):
    x0, y0, x1, y1 = box
    return ((x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * (1 + 2 * pad))


def crop(img, region, size):
    """Square crop (cx, cy, side) of an NCHW image, resampled to size x size; edges replicated."""
    cx, cy, side = region
    H, W = img.shape[2:]
    u = torch.linspace(-1, 1, size, device=img.device)
    X = (cx + u * side / 2) / (W - 1) * 2 - 1
    Y = (cy + u * side / 2) / (H - 1) * 2 - 1
    g = torch.stack(torch.meshgrid(Y, X, indexing="ij")[::-1], -1)[None].expand(img.shape[0], -1, -1, -1)
    return F.grid_sample(img, g, mode="bilinear", padding_mode="border", align_corners=True)


def uncrop(full, patch, mask, region):
    """Blend a crop (and its mask, both NCHW) back into the full image at region."""
    cx, cy, side = region
    H, W = full.shape[2:]
    xa, xb = max(0, int(math.floor(cx - side / 2))), min(W, int(math.ceil(cx + side / 2)) + 1)
    ya, yb = max(0, int(math.floor(cy - side / 2))), min(H, int(math.ceil(cy + side / 2)) + 1)
    if xb <= xa or yb <= ya:
        return full
    X = (torch.arange(xa, xb, dtype=full.dtype, device=full.device) - cx) / (side / 2)
    Y = (torch.arange(ya, yb, dtype=full.dtype, device=full.device) - cy) / (side / 2)
    g = torch.stack(torch.meshgrid(Y, X, indexing="ij")[::-1], -1)[None]
    p = F.grid_sample(patch, g.expand(patch.shape[0], -1, -1, -1), mode="bilinear", padding_mode="zeros",
                      align_corners=True)
    m = F.grid_sample(mask, g.expand(mask.shape[0], -1, -1, -1), mode="bilinear", padding_mode="zeros",
                      align_corners=True)
    out = full.clone()
    out[:, :, ya:yb, xa:xb] = full[:, :, ya:yb, xa:xb] * (1 - m) + p * m
    return out


def ident_grid(n, dev):
    u = torch.linspace(-1, 1, n, device=dev)
    return torch.stack(torch.meshgrid(u, u, indexing="ij")[::-1], -1)[None]  # (1, n, n, 2) x,y


def blur(x, s):
    """Separable Gaussian blur, computed on the GPU."""
    if s <= 0.3:
        return x
    dev0 = x.device
    x = x.to(DEV)
    r = max(1, int(3 * s))
    k = torch.exp(-torch.arange(-r, r + 1, dtype=x.dtype, device=x.device) ** 2 / (2 * s * s))
    k = (k / k.sum())
    c = x.shape[1]
    x = F.pad(x, (r, r, r, r), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(x, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c).to(dev0)


def dilate(x, r):
    """Square dilation by r px (separable max, on the GPU); erode = 1 - dilate(1 - x)."""
    r = int(r)
    if r < 1:
        return x
    dev0 = x.device
    x = x.to(DEV)
    x = F.max_pool2d(F.pad(x, (r, r, 0, 0), mode="replicate"), (1, 2 * r + 1), 1)
    x = F.max_pool2d(F.pad(x, (0, 0, r, r), mode="replicate"), (2 * r + 1, 1), 1)
    return x.to(dev0)


def erode(x, r):
    return 1 - dilate(1 - x, r)


# ------------------------------------------------------------------ models
def _get(name):
    if name in _M:
        return _M[name]
    if name == "dinov2":
        from transformers import AutoModel
        m = AutoModel.from_pretrained("facebook/dinov2-small")
    elif name == "clip":
        from transformers import CLIPVisionModelWithProjection
        m = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")
    elif name == "raft":
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        m = raft_large(weights=Raft_Large_Weights.DEFAULT)
    elif name == "sift":
        import kornia.feature as KF
        m = KF.SIFTFeature(num_features=2000, upright=False, device="cpu")
    elif name == "loftr":
        import kornia.feature as KF
        m = KF.LoFTR(pretrained="outdoor")
    elif name == "disk":
        import kornia.feature as KF
        m = KF.DISK.from_pretrained("depth")
    elif name == "lightglue":
        import kornia.feature as KF
        m = KF.LightGlueMatcher("disk")
    else:
        raise ValueError(name)
    _M[name] = m.eval()
    return _M[name]


class _on_gpu:
    def __init__(self, name):
        self.m = _get(name)

    def __enter__(self):
        return self.m.to(DEV)

    def __exit__(self, *a):
        self.m.to("cpu")
        if DEV == "cuda":
            torch.cuda.empty_cache()


_IMNET = (torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225]))
_CLIPN = (torch.tensor([0.4815, 0.4578, 0.4082]), torch.tensor([0.2686, 0.2613, 0.2758]))


def _norm(x, ms):
    return (x - ms[0].to(x).view(1, 3, 1, 1)) / ms[1].to(x).view(1, 3, 1, 1)


@torch.no_grad()
def patch_feats(x, kind, res):
    """Dense ViT patch features (N, C, h, w), L2-normalised. DINOv2: last layer; CLIP: penultimate."""
    if kind == "dinov2":
        res = res // 14 * 14
        with _on_gpu("dinov2") as m:
            xi = _norm(F.interpolate(x, size=(res, res), mode="bicubic", align_corners=False).clamp(0, 1), _IMNET)
            h = m(pixel_values=xi.to(DEV)).last_hidden_state[:, 1:]
        n = res // 14
    else:
        res = res // 16 * 16
        with _on_gpu("clip") as m:
            xi = _norm(F.interpolate(x, size=(res, res), mode="bicubic", align_corners=False).clamp(0, 1), _CLIPN)
            h = m(pixel_values=xi.to(DEV), interpolate_pos_encoding=True,
                  output_hidden_states=True).hidden_states[-2][:, 1:]
        n = res // 16
    f = h.float().cpu().transpose(1, 2).reshape(x.shape[0], -1, n, n)
    return F.normalize(f, dim=1)


@torch.no_grad()
def global_sim(a, b):
    """Cosine similarity of whole-image embeddings: DINOv2 CLS and CLIP image embedding."""
    out = {}
    with _on_gpu("dinov2") as m:
        f = lambda x: m(pixel_values=_norm(F.interpolate(x, size=(224, 224), mode="bicubic",
                                                         align_corners=False).clamp(0, 1), _IMNET).to(DEV)).pooler_output
        out["dino"] = F.cosine_similarity(f(a), f(b)).item()
    with _on_gpu("clip") as m:
        f = lambda x: m(pixel_values=_norm(F.interpolate(x, size=(224, 224), mode="bicubic",
                                                         align_corners=False).clamp(0, 1), _CLIPN).to(DEV)).image_embeds
        out["clip"] = F.cosine_similarity(f(a), f(b)).item()
    return out


# ------------------------------------------------------------------ correspondences
def _gray(x):
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])


def _px2n(p, n):
    return p / (n - 1) * 2 - 1


@torch.no_grad()
def match_semantic(Fa, Fb, fga, fgb, max_shift=0.5):
    """Mutual nearest neighbours of patch features (painting a -> photo b), positions in [-1, 1]."""
    n = Fa.shape[-1]
    pos = ident_grid(n, "cpu")[0].reshape(-1, 2)
    A = Fa[0].reshape(Fa.shape[1], -1)
    B = Fb[0].reshape(Fb.shape[1], -1)
    ia = torch.nonzero(F.interpolate(fga, size=(n, n), mode="area")[0, 0].reshape(-1) > 0.5).squeeze(1)
    ib = torch.nonzero(F.interpolate(fgb, size=(n, n), mode="area")[0, 0].reshape(-1) > 0.5).squeeze(1)
    if len(ia) < 8 or len(ib) < 8:
        return None, None, None
    S = A[:, ia].T @ B[:, ib]
    far = torch.cdist(pos[ia], pos[ib]) > max_shift * 2
    S = S.masked_fill(far, -2)
    j = S.argmax(1)
    i_back = S.argmax(0)
    mutual = i_back[j] == torch.arange(len(ia))
    s = S[torch.arange(len(ia)), j]
    keep = mutual & (s > 0.2)
    return pos[ia[keep]], pos[ib[j[keep]]], s[keep]


@torch.no_grad()
def match_keypoints(a, b, kind, size, max_shift=0.5):
    """Sparse keypoint matches painting a -> photo b (NCHW crops, same size), positions in [-1, 1]."""
    import kornia.feature as KF
    ga, gb = _gray(a), _gray(b)
    if kind == "sift":
        with _on_gpu("sift") as m:
            la, _, da = m(ga.to(DEV))
            lb, _, db = m(gb.to(DEV))
        _, idx = KF.match_smnn(da[0], db[0], 0.95)
        pa = KF.get_laf_center(la)[0][idx[:, 0]].cpu()
        pb = KF.get_laf_center(lb)[0][idx[:, 1]].cpu()
        conf = torch.ones(len(pa))
    elif kind == "loftr":
        with _on_gpu("loftr") as m:
            r = m({"image0": ga.to(DEV), "image1": gb.to(DEV)})
        pa, pb, conf = r["keypoints0"].cpu(), r["keypoints1"].cpu(), r["confidence"].cpu()
    elif kind == "disk+lightglue":
        with _on_gpu("disk") as d:
            fa = d(a.to(DEV), n=2048, pad_if_not_divisible=True)[0]
            fb = d(b.to(DEV), n=2048, pad_if_not_divisible=True)[0]
        with _on_gpu("lightglue") as lg:
            ka, kb = fa.keypoints, fb.keypoints
            lafa = KF.laf_from_center_scale_ori(ka[None], torch.ones(1, len(ka), 1, 1, device=DEV))
            lafb = KF.laf_from_center_scale_ori(kb[None], torch.ones(1, len(kb), 1, 1, device=DEV))
            dist, idx = lg(fa.descriptors, fb.descriptors, lafa, lafb, hw1=a.shape[2:], hw2=b.shape[2:])
        pa, pb = ka[idx[:, 0]].cpu(), kb[idx[:, 1]].cpu()
        conf = (1 - dist.squeeze(-1).cpu()).clamp(0, 1)
    else:
        raise ValueError(kind)
    pa, pb = _px2n(pa.float(), size), _px2n(pb.float(), size)
    keep = (pa - pb).norm(dim=1) < max_shift * 2
    return pa[keep], pb[keep], conf[keep]


def fit_affine(pa, pb, w=None, iters=600, thr=0.04, seed=0):
    """RANSAC affine pa -> pb (both [-1, 1]); returns (2x3, inlier mask) or (None, None)."""
    n = len(pa)
    if n < 6:
        return None, None
    g = torch.Generator().manual_seed(seed)
    idx = torch.stack([torch.randperm(n, generator=g)[:3] for _ in range(iters)])
    X = torch.cat([pa, torch.ones(n, 1)], 1)  # n x 3
    Xs = X[idx]  # it x 3 x 3
    Ys = pb[idx]  # it x 3 x 2
    ok = torch.linalg.det(Xs).abs() > 1e-4
    Xs, Ys = Xs[ok], Ys[ok]
    if len(Xs) == 0:
        return None, None
    M = torch.linalg.solve(Xs, Ys)  # it x 3 x 2
    res = ((X[None] @ M) - pb[None]).norm(dim=2)  # it x n
    inl = res < thr
    score = (inl.float() * (w[None] if w is not None else 1)).sum(1)
    det = torch.linalg.det(M[:, :2, :2])
    score[(det < 0.35) | (det > 3.0)] = -1
    best = score.argmax()
    if score[best] < max(6, 0.15 * n):
        return None, None
    m = inl[best]
    sol = torch.linalg.lstsq(X[m], pb[m]).solution  # 3 x 2
    d = torch.linalg.det(sol[:2, :2])
    if not (0.35 < d < 3.0):
        return None, None
    return sol.T, m  # 2 x 3


def affine_grid(A, n):
    g = ident_grid(n, "cpu")
    return g @ A[:, :2].T + A[:, 2]


@torch.no_grad()
def relocate(P, Q, P_reg, Q_reg, scales=(1.0, 0.8, 0.64, 0.51, 0.41, 0.33, 0.26), step=0.05, prefer=0.03):
    """The photo's head box can be far off (Qwen3-VL sometimes boxes the whole body, and the crop then
    shows a tiny head - no transform can bridge that). Search the photo region for the window that looks
    most like the painted head (DINOv2 CLS) over scales and shifts; keep the box unless a window is
    clearly (`prefer`) better. Returns (region, sim, sim_of_original_box)."""
    cx, cy, side = Q_reg
    cands = [(cx, cy, side)]
    for s in scales[1:]:
        lim = 0.5 - s / 2 + 1e-6
        k = int(lim / step)
        offs = [i * step for i in range(-k, k + 1)]
        cands += [(cx + dx * side, cy + dy * side, side * s) for dy in offs for dx in offs]
    a = crop(P, P_reg, 224)
    es = []
    with _on_gpu("dinov2") as m:
        f = lambda x: F.normalize(m(pixel_values=_norm(x, _IMNET).to(DEV)).pooler_output.float().cpu(), dim=1)  # noqa: E731
        ea = f(a)
        for i in range(0, len(cands), 64):
            es.append(f(torch.cat([crop(Q, r, 224) for r in cands[i:i + 64]])))
    sim = (torch.cat(es) @ ea[0])
    i = int(sim.argmax())
    if sim[i] < sim[0] + prefer:
        i = 0
    return cands[i], float(sim[i]), float(sim[0])


ANCHOR_SHIFT = 0.2  # max displacement of a match after the landmark pre-alignment (crop units)
ANCHOR_TOL = 0.15  # the fit may move a landmark at most this share of the face size


def landmark_anchors(plm, qlm, P_reg, Q_reg, A0):
    """Eyes + nose as anchor pairs in the pre-aligned frame: painting positions, and the photo positions
    taken back through A0. Also the face size (crop units) for the gate tolerance."""
    keys = ("eye_a", "eye_b", "nose")
    to = lambda lm, r: torch.tensor([[(lm[k][0] - r[0]) / (r[2] / 2), (lm[k][1] - r[1]) / (r[2] / 2)] for k in keys])  # noqa: E731
    a, b = to(plm, P_reg), to(qlm, Q_reg)
    m = tf.GLOBAL["similarity"]
    inv = m.make(m.fit(A0(a)[None], a[None], torch.ones(1, 3))[0])  # A0 is a similarity: exact inverse
    face = max(float((a[1] - a[0]).norm()), float((a[2] - (a[0] + a[1]) / 2).norm()))
    return a, inv(b), face


def landmark_init(plm, qlm, P_reg, Q_reg):
    """Similarity painting crop -> photo crop from the eyes and the nose (pixels -> crop coords), and
    its mean residual in % of the crop side (large = the head is turned differently)."""
    keys = ("eye_a", "eye_b", "nose")
    to = lambda lm, r: torch.tensor([[(lm[k][0] - r[0]) / (r[2] / 2), (lm[k][1] - r[1]) / (r[2] / 2)] for k in keys])  # noqa: E731
    a, b = to(plm, P_reg), to(qlm, Q_reg)
    m = tf.GLOBAL["similarity"]
    Pm = m.fit(a[None], b[None], torch.ones(1, 3))[0]
    T = m.make(Pm)
    return T, float((T(a) - b).norm(dim=1).mean() / 2 * 100)


def check_points(fg, n=13):
    """Coarse grid points on (and just around) the animal: there a fitted transform must not fold
    or stretch wildly (polynomials and homographies may do anything outside the matches)."""
    u = torch.linspace(-1, 1, n)
    pts = torch.stack(torch.meshgrid(u, u, indexing="ij")[::-1], -1).reshape(-1, 2)
    v = F.grid_sample(dilate(fg, 8), pts.view(1, 1, -1, 2), align_corners=True)[0, 0, 0] > 0.5
    return pts[v]


def fit_anchored(spec, pa, pb, w, thr, fg, anc_a, anc_b, tol, anchor_weight=0.15):
    """fit_transform with landmark anchors (eyes, nose; tensors (k, 2), same frame as the matches):
    each anchor joins the matches with anchor_weight x the total match weight (RANSAC prefers models
    that agree with them, TPS nearly interpolates them); then a gate: if the fit still moves an anchor
    more than `tol` away, the matches are not trusted (neighbour, body, turned head) and the identity
    (= the landmark pre-alignment) is used. Returns (T, inliers of the matches or None, info)."""
    n = len(pa)
    k = len(anc_a)
    wa = torch.full((k,), float(w.sum()) * anchor_weight)
    PA, PB, WW = torch.cat([pa, anc_a]), torch.cat([pb, anc_b]), torch.cat([w, wa])
    T, inl = fit_transform(spec, PA, PB, WW, thr, fg, force=torch.arange(n, n + k))
    info = {}
    if T is not None:
        err = float((T(anc_a) - anc_b).norm(dim=1).max())
        info["anchor_err"] = round(err, 4)
        if err > tol:
            T = None
            info["fallback"] = "fit moved landmarks by %.3f > %.3f" % (err, tol)
    if T is None:
        info.setdefault("fallback", "no fit")
        return (lambda x: x.clone()), None, info
    return T, inl[:n], info


def fit_transform(spec, pa, pb, w, thr, fg, force=None):
    """spec: "legacy" (the old RANSAC affine) or "MODEL/ESTIMATOR[+tps(lam=..)|+ffd(grid=..,lam=..)]",
    e.g. "homography/ransac+ffd(grid=6,lam=1)". Returns (callable painting -> photo, inliers) or (None, None).
    force: indices always passed to the non-rigid stage (anchors), whether inliers of the global fit or not."""
    if spec == "legacy":
        A, inl = fit_affine(pa, pb, w, thr=thr)
        if A is None:
            return None, None
        return (lambda x: x @ A[:, :2].T + A[:, 2]), inl
    base, nr = re.match(r"^(.*?)(?:\+(tps\(.*\)|ffd\(.*\)))?$", spec).groups()
    g, est = base.split("/")
    T, inl = tf.fit_global(g, est, pa, pb, w, thr=thr, check=check_points(fg))
    if T is None or not nr:
        return T, inl
    sel = inl.clone()
    if force is not None:
        sel[force] = True
    kind, params = nr.rstrip(")").split("(")
    kw = {k: float(v) for k, v in (kv.split("=") for kv in params.split(","))}
    if kind == "tps":
        keep = None
        if force is not None:  # positions of the anchors inside the selected subset
            keep = torch.nonzero(torch.isin(torch.nonzero(sel)[:, 0], force))[:, 0]
        return tf.fit_tps(T, pa[sel], pb[sel], w[sel], keep=keep, **kw), inl
    kw["grid"] = int(kw["grid"])
    return tf.fit_ffd(T, pa[sel], pb[sel], w[sel], **kw), inl


@torch.no_grad()
def raft_refine(a, b_warped, size=512):
    """Dense residual flow (normalised units) taking painting a to the aligned photo, + FB error."""
    with _on_gpu("raft") as m:
        sm = lambda x: (F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False) * 2 - 1).to(DEV)
        fw = m(sm(a), sm(b_warped), num_flow_updates=20)[-1].float().cpu()
        bw = m(sm(b_warped), sm(a), num_flow_updates=20)[-1].float().cpu()
    fw = blur(fw, 2.0)
    g = ident_grid(size, "cpu")
    back = F.grid_sample(bw, g + fw.permute(0, 2, 3, 1) * 2 / (size - 1), align_corners=True, padding_mode="border")
    err = (fw + back).norm(dim=1, keepdim=True) / size  # share of the crop
    return fw.permute(0, 2, 3, 1) * 2 / (size - 1), err  # (1, s, s, 2), (1, 1, s, s)


@torch.no_grad()
def dense_semantic(Fa, Fb_warped, radius=2):
    """Per-patch nearest neighbour of the painting feature in a small window of the aligned photo."""
    n = Fa.shape[-1]
    k = 2 * radius + 1
    Bu = F.unfold(F.pad(Fb_warped, (radius,) * 4, mode="replicate"), k)  # 1, C*k*k, n*n
    Bu = Bu.view(1, Fa.shape[1], k * k, n * n)
    s = (Fa.reshape(1, Fa.shape[1], 1, n * n) * Bu).sum(1)  # 1, k*k, n*n
    w = torch.softmax(s / 0.03, 1)
    off = torch.stack(torch.meshgrid(torch.arange(-radius, radius + 1), torch.arange(-radius, radius + 1),
                                     indexing="ij")[::-1], -1).reshape(-1, 2).float()  # k*k x 2 (x, y)
    d = (w.permute(0, 2, 1) @ off).reshape(1, n, n, 2) * 2 / (n - 1)
    return blur(d.permute(0, 3, 1, 2), 1.0).permute(0, 2, 3, 1)


# ------------------------------------------------------------------ comparison
def _lab(x):
    import kornia.color as KC
    return KC.rgb_to_lab(x.clamp(0, 1))


def heat(v):
    """0..1 -> RGB (dark blue, cyan, yellow, red)."""
    v = v.clamp(0, 1)
    r = (v * 2 - 0.6).clamp(0, 1)
    g = (1 - (v * 2 - 1).abs() * 1.2).clamp(0, 1)
    b = (1 - v * 2.2).clamp(0, 1) * 0.9 + 0.1
    return torch.cat([r, g, b], 1)


def dots(img, pts, color, r=2):
    img = img.clone()
    n = img.shape[-1]
    for p in pts:
        x = int(round((p[0].item() + 1) / 2 * (n - 1)))
        y = int(round((p[1].item() + 1) / 2 * (n - 1)))
        img[0, :, max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = torch.tensor(color).view(3, 1, 1)
    return img


@torch.no_grad()
def analyse(painting, photo, painting_box, photo_box, method="dinov2+raft", compare="dinov2",
            sensitivity=2.0, max_area=0.18, min_blob=0.002, pad=0.15, out_size=1024, work=512, sem_res=644,
            restrict_box=None, min_inliers=0.55, min_flow=0.35, min_aligned=0.75,
            painting_others=(), photo_others=(), transform="legacy", painting_lm=None, photo_lm=None,
            init="none", max_lm_resid=6.0):
    """painting, photo: NCHW 0..1 (CPU). Returns dict with crops, mask, debug image, report.
    *_others: face boxes of the other animals in the same image, cut out of the comparison.
    transform: geometric model fitted to the matches, see fit_transform / TRANSFORMS."""
    painting_box = painting_box or fallback_box(painting)
    photo_box = photo_box or fallback_box(photo)
    P_reg, Q_reg = region_of(painting_box, pad), region_of(photo_box, pad)
    Pc, Qc = crop(painting, P_reg, work), crop(photo, Q_reg, work)
    fgP, fgQ = fg_mask(Pc), fg_mask(Qc)
    ownP = own_mask(Pc, P_reg, painting_box, painting_others, fgP, sem_res)
    ownQ = own_mask(Qc, Q_reg, photo_box, photo_others, fgQ, sem_res)
    fgP, fgQ = fgP * ownP, fgQ * ownQ
    rep = {"method": method, "transform": transform}

    # --- 0. initial alignment A0 (painting crop -> photo crop): the photo is resampled from the full
    # image into the painting's frame, so the matches below see both heads at the same scale and angle
    H, W = photo.shape[2:]
    qcx, qcy, qside = Q_reg

    def sampleQ(g, src=photo):  # photo-crop coords -> full-resolution photo
        gx = (qcx + g[..., 0] * qside / 2) / (W - 1) * 2 - 1
        gy = (qcy + g[..., 1] * qside / 2) / (H - 1) * 2 - 1
        return F.grid_sample(src, torch.stack([gx, gy], -1), align_corners=True, padding_mode="border")

    A0 = None
    if init in ("auto", "landmarks") and painting_lm and photo_lm:
        A0, resid = landmark_init(painting_lm, photo_lm, P_reg, Q_reg)
        rep["init"], rep["lm_resid"] = "landmarks", round(resid, 2)
    elif init in ("auto", "reloc"):
        (ncx, ncy, ns), s1, s0 = relocate(painting, photo, P_reg, Q_reg)
        if (ncx, ncy, ns) != tuple(Q_reg):
            off = torch.tensor([(ncx - qcx) / (qside / 2), (ncy - qcy) / (qside / 2)])
            k = ns / qside
            A0 = lambda x: off + x * k  # noqa: E731
            rep["init"], rep["reloc"] = "reloc", [round(k, 3), round(s1 - s0, 3)]
    if A0 is not None:
        g0 = A0(ident_grid(work, "cpu"))
        Qc = sampleQ(g0)
        ownQ = (F.grid_sample(ownQ, g0, align_corners=True, padding_mode="zeros") > 0.5).float()
        fgQ = fg_mask(Qc) * ownQ
    else:
        A0 = lambda x: x  # noqa: E731

    # --- 1. global alignment painting-crop -> photo-crop (normalised coords)
    T = None
    base, refine = method.split("+")[0], (method.split("+")[1] if "+" in method else None)
    if method == "disk+lightglue":
        base, refine = "disk+lightglue", None
    pa = pb = None
    if base in ("dinov2", "clip"):
        Fa, Fb = patch_feats(Pc, base, sem_res), patch_feats(Qc, base, sem_res)
        anchors = rep.get("init") == "landmarks" and transform != "legacy"
        # after the landmark pre-alignment a true match moves little: far matches are the neighbour/body
        pa, pb, w = match_semantic(Fa, Fb, fgP, fgQ, max_shift=ANCHOR_SHIFT if anchors else 0.5)
        rep["matches"] = 0 if pa is None else len(pa)
        thr = 2.5 / Fa.shape[-1] * 2
        if pa is not None and anchors:
            anc_a, anc_b, face = landmark_anchors(painting_lm, photo_lm, P_reg, Q_reg, A0)
            T2, inl, info = fit_anchored(transform, pa, pb, w, thr, fgP, anc_a, anc_b, tol=ANCHOR_TOL * face)
            rep.update(info)
            T = T2
            if inl is not None:
                pa, pb = pa[inl], pb[inl]
                rep["inliers"] = int(inl.sum())
            else:  # landmark pre-alignment only: the matches are not used, so they cannot veto
                rep["inliers"] = rep["matches"]
        elif pa is not None:
            T2, inl = fit_transform(transform, pa, pb, w, thr, fgP)
            if T2 is not None:
                T, pa, pb = T2, pa[inl], pb[inl]
                rep["inliers"] = int(inl.sum())
    elif base in ("sift", "loftr", "disk+lightglue"):
        try:
            pa, pb, w = match_keypoints(Pc, Qc, base, work)
            rep["matches"] = len(pa)
            T2, inl = fit_transform(transform, pa, pb, w, 0.03, fgP)
            if T2 is not None:
                T, pa, pb = T2, pa[inl], pb[inl]
                rep["inliers"] = int(inl.sum())
        except Exception as e:  # weights unavailable, too few points ...
            rep["error"] = "%s: %s" % (type(e).__name__, str(e)[:120])
            pa = pb = None
    rep["affine"] = "inliers" in rep
    # no transform = identity grid; the safety gate below then skips the head ("no consistent matches")
    grid = tf.grid_of(T, work) if T is not None else ident_grid(work, "cpu")
    Qw = F.grid_sample(Qc, grid, align_corners=True, padding_mode="border")

    # --- 2. local refinement
    valid = F.grid_sample(fgQ, grid, align_corners=True, padding_mode="zeros")
    if refine == "raft" or method == "raft":
        d, err = raft_refine(Pc, Qw, work)
        grid = F.grid_sample(grid.permute(0, 3, 1, 2), ident_grid(work, "cpu") + d, align_corners=True,
                             padding_mode="border").permute(0, 2, 3, 1)
        rel = (err < 0.02).float()
        rep["flow_reliable"] = round(float((rel * fgP).sum() / fgP.sum().clamp(min=1)), 3)
        valid = F.grid_sample(fgQ, grid, align_corners=True, padding_mode="zeros") * blur(rel, 3)
    elif refine == "dense":
        Fa = patch_feats(Pc, "dinov2", sem_res)
        Fbw = patch_feats(F.grid_sample(Qc, grid, align_corners=True, padding_mode="border"), "dinov2", sem_res)
        d = dense_semantic(Fa, Fbw)
        d = F.interpolate(d.permute(0, 3, 1, 2), size=(work, work), mode="bilinear", align_corners=True)
        grid = F.grid_sample(grid.permute(0, 3, 1, 2), ident_grid(work, "cpu") + d.permute(0, 2, 3, 1),
                             align_corners=True, padding_mode="border").permute(0, 2, 3, 1)
        valid = F.grid_sample(fgQ, grid, align_corners=True, padding_mode="zeros")
    Qw = F.grid_sample(Qc, grid, align_corners=True, padding_mode="border")

    # --- 3. where do they disagree?
    Fp, Fq = patch_feats(Pc, compare, sem_res), patch_feats(Qw, compare, sem_res)
    sem = 1 - (Fp * Fq).sum(1, keepdim=True)
    sem = F.interpolate(sem, size=(work, work), mode="bicubic", align_corners=False).clamp(0, 2)
    # the outline itself is the painter's business (ears, fur wisps): look only inside the animal
    region = erode(((fgP * valid) > 0.5).float(), work / 40) > 0.5
    if restrict_box is not None:  # e.g. the face box: only there do we look for lost features
        rb = torch.zeros_like(fgP)
        x0, y0, x1, y1 = [int(round((v + 1) / 2 * (work - 1))) for v in restrict_box]
        rb[..., max(0, y0):y1, max(0, x0):x1] = 1
        region = region & (rb > 0.5)
    rep["aligned_sim"] = round(float(1 - sem[region].mean()) if region.any() else 0.0, 3)
    # --- safety gate: if the photo cannot be aligned to the painting reliably (head turned, several
    # animals, ...), every difference would be a misalignment, not a lost feature: change nothing
    why = []
    if "matches" in rep and "inliers" not in rep:
        why.append("no consistent matches")
    if "inliers" in rep and rep["inliers"] < min_inliers * rep["matches"]:
        why.append("inliers %d/%d" % (rep["inliers"], rep["matches"]))
    if "flow_reliable" in rep and rep["flow_reliable"] < min_flow:
        why.append("flow reliable %.2f" % rep["flow_reliable"])
    if rep["aligned_sim"] < min_aligned:
        why.append("aligned sim %.2f" % rep["aligned_sim"])
    if rep.get("lm_resid", 0) > max_lm_resid:  # eyes + nose do not fit one similarity: head turned
        why.append("landmarks %.1f%%" % rep["lm_resid"])
    if why:
        rep["skipped"] = "pose differs: " + ", ".join(why)
        region = region & False
    s = work / 256  # fine enough for nose freckles
    La, Lb = _lab(blur(Pc, s)), _lab(blur(Qw, s))
    # lighting is not a feature: compare lightness only above the scale of shading
    shade = work / 16
    La[:, :1] = La[:, :1] - blur(La[:, :1], shade)
    Lb[:, :1] = Lb[:, :1] - blur(Lb[:, :1], shade)
    # global colour drift is not a lost feature: compare after matching mean/std inside the animal
    m = region.float()
    if m.sum() > 50:
        mu = lambda x: (x * m).sum((2, 3), keepdim=True) / m.sum()
        sd = lambda x, u: (((x - u) ** 2 * m).sum((2, 3), keepdim=True) / m.sum()).sqrt() + 1e-3
        ua, ub = mu(La), mu(Lb)
        Lb = (Lb - ub) / sd(Lb, ub) * sd(La, ua) + ua
    app = ((La - Lb) * torch.tensor([1.5, 1.0, 1.0]).view(1, 3, 1, 1)).norm(dim=1, keepdim=True) / 100
    med = lambda x: x[region].median().clamp(min=1e-4) if region.any() else torch.tensor(1.0)
    # appearance decides (small marks are semantically weak), semantics up- or down-weights it:
    # a region where the ViT features still agree is a style difference, not a lost feature
    sem_w = (sem / med(sem)).clamp(0.5, 2.5).sqrt()
    score = (app / med(app)) * sem_w * region.float()
    score = blur(score, work / 256)

    def clean(m):  # drop specks: an opening by ~1 % of the crop, then a slight closing
        r = max(1, int(work / 128))
        return dilate(erode(dilate(m, r // 2), r), r // 2 + r) * region.float()

    mask = clean((score > sensitivity).float())
    area = float(mask.sum() / region.float().sum().clamp(min=1))
    if area > max_area and region.any():  # keep only the strongest disagreements
        v = score[region].float()
        t = torch.quantile(v[torch.randperm(len(v))[:200000]], 1 - max_area)
        mask = clean((score > max(t.item(), sensitivity)).float())
        area = float(mask.sum() / region.float().sum().clamp(min=1))
    if area < min_blob:
        mask.zero_()
        area = 0.0
    rep["mask_area"] = round(area, 3)

    # --- 4. outputs at out_size: painting crop with the photo features pasted in
    Pc_o = crop(painting, P_reg, out_size)
    grid_o = F.interpolate(grid.permute(0, 3, 1, 2), size=(out_size, out_size), mode="bilinear",
                           align_corners=True).permute(0, 2, 3, 1)
    Qw_o = sampleQ(A0(grid_o))  # straight from the full-resolution photo, through A0 and the fit
    hard = F.interpolate(mask, size=(out_size, out_size), mode="bilinear", align_corners=False)
    grow = dilate(hard, out_size / 64)
    band = (dilate(grow, out_size / 24) - grow).clamp(0, 1)
    band = band * F.interpolate(fgP, size=(out_size, out_size), mode="bilinear", align_corners=False)
    Qa = Qw_o
    if band.sum() > 100:  # colours of the photo feature -> palette of the painting around it
        mu = lambda x: (x * band).sum((2, 3), keepdim=True) / band.sum()
        sd = lambda x, u: (((x - u) ** 2 * band).sum((2, 3), keepdim=True) / band.sum()).sqrt() + 1e-3
        up, uq = mu(Pc_o), mu(Qw_o)
        Qa = ((Qw_o - uq) / sd(Qw_o, uq) * sd(Pc_o, up) + up).clamp(0, 1)
    soft = blur(grow, out_size / 96).clamp(0, 1)
    pasted = Pc_o * (1 - soft) + Qa * soft
    inpaint_mask = blur(dilate(hard, out_size / 32), out_size / 48).clamp(0, 1)
    # never repaint the neighbour: stay on this animal (a little outside its outline for the fur)
    own_o = dilate(F.interpolate(ownP, size=(out_size, out_size), mode="bilinear", align_corners=False), out_size / 64)
    inpaint_mask = inpaint_mask * blur(own_o, out_size / 128)
    soft = soft * own_o
    pasted = Pc_o * (1 - soft) + Qa * soft

    # --- debug strip
    t = 320
    rs = lambda x: F.interpolate(x, size=(t, t), mode="bilinear", align_corners=False)
    pa_img, pb_img = rs(Pc * (0.3 + 0.7 * ownP)), rs(Qc * (0.3 + 0.7 * ownQ))  # neighbours dimmed
    if pa is not None and len(pa):
        sel = torch.randperm(len(pa))[:150]
        pa_img = dots(pa_img, pa[sel], (1, 0.1, 0.1))
        pb_img = dots(pb_img, pb[sel], (1, 0.1, 0.1))
    ov = rs(Pc) * (1 - 0.55 * rs(mask)) + torch.tensor([1, 0, 0.6]).view(1, 3, 1, 1) * 0.55 * rs(mask)
    tiles = [pa_img, pb_img, rs(Qw), heat(rs(sem / (2 * med(sem)))), heat(rs(app / (3 * med(app)))),
             heat(rs(score / (2 * sensitivity))), ov, rs(pasted)]
    debug = torch.cat([torch.cat([x, torch.ones(1, 3, t, 4)], 3) for x in tiles], 3)
    return dict(painting_crop=Pc_o, aligned_photo=Qw_o, pasted=pasted, inpaint_mask=inpaint_mask,
                mask=hard, debug=debug, region=P_reg, photo_region=Q_reg, report=rep)
