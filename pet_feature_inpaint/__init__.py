"""EXPERIMENTAL nodes (on top of the t6 workflow, which does not use them):
Pet Feature Match - aligns the photo to the painted portrait (keypoint trackers, RAFT, DINOv2 / CLIP
    patch features), finds features of the photo that the painting lost and pastes them in;
Pet Feature Uncrop - puts the repainted head crop back into the portrait;
Pet Likeness Score - DINOv2 / CLIP similarity of the painted head and the photo head;
Pet Grounded Layout (grounded.py) - one or several animals, each bust always starts at the bottom edge.
The work is done in featcore.py, reloaded on every run (experiment: edit and re-queue, no restart)."""
import importlib
import json
import os

import torch

from . import featcore

_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "featcore.py")


def _core():
    global featcore
    from . import transforms
    importlib.reload(transforms)
    featcore = importlib.reload(featcore)
    return featcore


def _nchw(x):
    return x[:1].permute(0, 3, 1, 2).float().cpu()


def _nhwc(x, like):
    return x.permute(0, 2, 3, 1).clamp(0, 1).to(like.device, like.dtype)


class PetFeatureMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "painting": ("IMAGE",), "photo": ("IMAGE",),
            "painting_head": ("STRING", {"forceInput": True, "tooltip": 'Qwen3-VL "bbox_2d" of the head on the painting.'}),
            "photo_head": ("STRING", {"forceInput": True, "tooltip": 'Qwen3-VL "bbox_2d" of the head on the photo.'}),
            "method": (featcore.METHODS, {"default": "dinov2+raft",
                                          "tooltip": "How the photo is aligned to the painting."}),
            "compare": (["dinov2", "clip"], {"default": "dinov2", "tooltip": "Features used to find lost details."}),
            "sensitivity": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 6.0, "step": 0.1,
                                      "tooltip": "Lower = more regions count as lost."}),
            "max_area": ("FLOAT", {"default": 0.18, "min": 0.01, "max": 0.6, "step": 0.01,
                                   "tooltip": "At most this share of the head is repainted."}),
            "size": ("INT", {"default": 1024, "min": 512, "max": 1536, "step": 64}),
            "min_aligned": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "Safety: below this DINOv2 similarity after alignment the pose "
                                                 "differs too much and the head is left as painted."}),
        }, "optional": {
            "transform": (featcore.TRANSFORMS, {"default": featcore.DEFAULT_TRANSFORM,
                                                "tooltip": "Geometric model + robust estimator that maps the photo "
                                                           "onto the painting (legacy = old RANSAC affine)."}),
            "init": (["auto", "landmarks", "reloc", "none"], {
                "default": "auto", "tooltip": "Pre-alignment before the matches. landmarks: similarity from the "
                "eyes + nose (needs FIND_LANDMARKS answers); reloc: DINOv2 search for the head on the photo; "
                "auto: landmarks when given, else reloc."}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("pasted_crop", "inpaint_mask", "debug", "region", "report", "painting_crop")
    FUNCTION = "run"
    CATEGORY = "pet_portrait/experimental"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return os.path.getmtime(_CORE)

    def run(self, painting, photo, painting_head, photo_head, method, compare, sensitivity, max_area, size,
            min_aligned=0.75, transform="legacy", init="none"):
        fc = _core()
        P, Q = _nchw(painting), _nchw(photo)
        r = fc.analyse_all(P, Q, painting_head, photo_head, method=method, compare=compare,
                           sensitivity=sensitivity, max_area=max_area, out_size=size, min_aligned=min_aligned,
                           transform=transform, init=init)
        text = json.dumps(r["reports"], ensure_ascii=False)
        print("[PetFeatureMatch]", text)
        region = json.dumps([{"cx": cx, "cy": cy, "side": s} for cx, cy, s in r["regions"]])
        return (_nhwc(r["pasted"], painting), r["inpaint_mask"][:, 0].to(painting.device),
                _nhwc(r["debug"], painting), region, text, _nhwc(r["painting_crop"], painting))


class PetFeatureUncrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"painting": ("IMAGE",), "patch": ("IMAGE",), "mask": ("MASK",),
                             "region": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "run"
    CATEGORY = "pet_portrait/experimental"

    def run(self, painting, patch, mask, region):
        fc = _core()
        regions = json.loads(region)
        if isinstance(regions, dict):
            regions = [regions]
        out = _nchw(painting)
        full = torch.zeros_like(out[:, :1])  # where the crops changed the portrait (full frame)
        X = patch.permute(0, 3, 1, 2).float().cpu()
        M = (mask if mask.dim() == 3 else mask[:, 0]).float().cpu()[:, None]
        if X.shape[2:] != M.shape[2:]:
            M = torch.nn.functional.interpolate(M, size=X.shape[2:], mode="bilinear", align_corners=False)
        for i, r in enumerate(regions):  # one head crop per animal
            reg = (r["cx"], r["cy"], r["side"])
            out = fc.uncrop(out, X[i:i + 1], M[i:i + 1], reg)
            full = fc.uncrop(full, torch.ones_like(M[i:i + 1]), M[i:i + 1], reg)
        return (_nhwc(out, painting), full[:, 0].clamp(0, 1).to(painting.device))


class PetStrengthMerge:
    """Strength mask for Differential Diffusion: max(base, extra * extra_strength). Used so the final
    SDXL pass repaints pasted photo features more firmly than the protected face around them."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"base": ("MASK",), "extra": ("MASK",),
                             "extra_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05})}}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "run"
    CATEGORY = "pet_portrait/experimental"

    def run(self, base, extra, extra_strength):
        b = base if base.dim() == 3 else base[:, 0]
        e = (extra if extra.dim() == 3 else extra[:, 0]).to(b)
        if e.shape[1:] != b.shape[1:]:
            e = torch.nn.functional.interpolate(e[:, None], size=b.shape[1:], mode="bilinear", align_corners=False)[:, 0]
        if len(e) != len(b):
            e = e[:1].expand(len(b), -1, -1)
        return (torch.maximum(b, e.clamp(0, 1) * extra_strength),)


class PetLikenessScore:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"painting": ("IMAGE",), "photo": ("IMAGE",),
                             "painting_head": ("STRING", {"forceInput": True}),
                             "photo_head": ("STRING", {"forceInput": True}),
                             "label": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "pet_portrait/experimental"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kw):
        return os.path.getmtime(_CORE)

    def run(self, painting, photo, painting_head, photo_head, label):
        fc = _core()
        s, n = fc.likeness(_nchw(painting), _nchw(photo), painting_head, photo_head)
        text = json.dumps(dict(label=label, heads=n, **s))
        print("[PetLikenessScore]", text)
        return {"ui": {"text": [text]}, "result": (text,)}


from .grounded import PetGroundedLayout  # noqa: E402

NODE_CLASS_MAPPINGS = {"PetFeatureMatch": PetFeatureMatch, "PetFeatureUncrop": PetFeatureUncrop,
                       "PetLikenessScore": PetLikenessScore, "PetStrengthMerge": PetStrengthMerge,
                       "PetGroundedLayout": PetGroundedLayout}
NODE_DISPLAY_NAME_MAPPINGS = {"PetGroundedLayout": "Pet Grounded Layout (bust always on the bottom edge)",
                              "PetFeatureMatch": "Pet Feature Match (experimental)",
                              "PetFeatureUncrop": "Pet Feature Uncrop (experimental)",
                              "PetStrengthMerge": "Pet Strength Merge (experimental)",
                              "PetLikenessScore": "Pet Likeness Score (experimental)"}
