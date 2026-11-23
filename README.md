# Pet Portrait v2 — ComfyUI nodes and workflow

Photo → pastel painted pet portrait. One node pack with two parts:

* `pet_portrait_layout/` — layout, pastel palette, multi-animal split/group, face mask, coat colours,
  flat background (pure PyTorch);
* `pet_feature_inpaint/` (experimental) — restores features of the real animal that the generated
  portrait lost: heads from Qwen3-VL eye/nose points, photo aligned to the painting
  (homography/RANSAC + TPS), DINOv2 semantic × Lab appearance difference → mask, safety gate that skips
  the paste when the pose differs too much;
* `pet_feature_inpaint/grounded.py` — **Pet Grounded Layout**, the layout used by the workflow: every
  animal (alone or in a group, each placed on its own) starts at the bottom edge and never floats.
  Its solid bust, not a shadow, a stray tuft or a painted fade-out, runs 4 % past the edge. The brushy
  fade ends below the edge. Without a usable head box, the whole animal is fitted from `top_space`
  down to the edge.

Install: `git clone` this repo into `ComfyUI/custom_nodes/`, `pip install -r requirements.txt`,
restart ComfyUI. DINOv2-small and CLIP ViT-B/16 are downloaded from Hugging Face on first use.
Needs a recent ComfyUI (core `TextGenerate`, `RemoveBackground`, FLUX.2 nodes).

## Workflow

`workflows/pet_photo_portrait_feat2_workflow.json` (API format: `*_api.json`):
Qwen3-VL description → FLUX.2 klein → coat colours → grounded layout → nose freckles →
lost features pasted from the photo → SDXL painterly pass (always the last step) → save.

Models:

| file | folder |
|---|---|
| `flux-2-klein-4b-fp8.safetensors` | `diffusion_models` |
| `qwen_3_4b.safetensors` | `text_encoders` |
| `qwen3vl_4b_fp8_scaled.safetensors` | `text_encoders` |
| `flux2-vae.safetensors` | `vae` |
| `sd_xl_base_1.0.safetensors` | `checkpoints` |
| `birefnet.safetensors` | `background_removal` |
