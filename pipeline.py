"""
pipeline.py — Image devastation pipeline for the IBM Z hackathon web app.

Public interface:
    run(image_bytes: bytes, yaml_path: str) -> str

Takes raw image bytes, runs SegFormer semantic segmentation (CPU),
assembles a dynamic inpainting prompt, and submits a non-blocking
FLUX.1 Fill job to Replicate. Returns the prediction polling URL.

The caller is responsible for polling the returned URL for completion.
"""

import io
import os
import tempfile

import numpy as np
import replicate
import torch
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv
from PIL import Image, ImageFilter
from transformers import SegformerForSemanticSegmentation

# ---------------------------------------------------------------------------
# Load environment variables from .env at import time
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "starry12tree/segformer-b2-finetuned-ade-512-512"
REPLICATE_MODEL = "black-forest-labs/flux-fill-dev"
MAX_DIM = 1024

# ADE20K label keywords that map to impact mask value 255 (inpaint these areas)
IMPACT_KEYWORDS = [
    "water", "river", "sea", "lake", "waterfall",
    "grass", "field", "earth", "ground", "dirt", "soil", "sand",
    "rock", "stone", "mountain",
    "tree", "plant", "flora", "forest", "bush",
    "floor", "terrain", "land",
    "sky", "person",
]

# Coverage aggregation: semantic category -> ADE20K label substrings
COVERAGE_KEYWORDS = {
    "water_body": ["water", "river", "sea", "lake", "waterfall"],
    "turf_ground": ["grass", "field", "earth", "ground", "dirt",
                    "soil", "sand", "rock", "stone", "mountain"],
    "arboreal_canopy": ["tree", "plant", "flora", "forest", "bush"],
}

# ---------------------------------------------------------------------------
# Mask smoothing (pure numpy — no OpenCV dependency)
# ---------------------------------------------------------------------------
MASK_BLUR_KERNEL = 31

# Sigma for the mask softening. cv2 derives this from a 31x31 kernel with
# sigmaX=0; we pass the same sigma straight to Pillow instead of building a
# kernel. Pillow approximates the Gaussian with 3 successive box filters, so
# results drift up to ~17/255 from cv2 at hard edges — irrelevant here: the
# mask is binary 0/255 and FLUX thresholds it back anyway.
MASK_BLUR_SIGMA = 0.3 * ((MASK_BLUR_KERNEL - 1) * 0.5 - 1) + 0.8

# SegFormer input spec (ADE20K 512x512 fine-tuned): fixed square resize,
# ImageNet mean/std, NCHW float32 tensor.
SEG_INPUT_SIZE = 512
SEG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SEG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _gaussian_blur_2d(
    mask: np.ndarray, sigma: float = MASK_BLUR_SIGMA
) -> np.ndarray:
    """
    Soften a 2-D uint8 mask with a Gaussian blur.

    Implemented with Pillow (already a hard dependency) rather than OpenCV or a
    hand-rolled convolution: ~24x faster than the numpy version on CPU, which
    matters because s390x has no vectorised fast path for this.

    Returns uint8 with the same H×W shape as the input.
    """
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {mask.shape}")

    image = Image.fromarray(np.ascontiguousarray(mask, dtype=np.uint8), mode="L")
    blurred = image.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.asarray(blurred)


# Prompt threshold bands per category: (lower_exclusive, upper_inclusive, band_key)
# upper value of 101 acts as open-ended upper bound (> lower means dominant)
THRESHOLDS = {
    "water_body":      [(0,  8,   "minor"), (8,  25,  "moderate"), (25, 101, "dominant")],
    "turf_ground":     [(0,  15,  "minor"), (15, 35,  "moderate"), (35, 101, "dominant")],
    "arboreal_canopy": [(0,  10,  "minor"), (10, 30,  "moderate"), (30, 101, "dominant")],
}

# ---------------------------------------------------------------------------
# Lazy singleton — loaded once on first run() call, reused across requests
# ---------------------------------------------------------------------------
_model: SegformerForSemanticSegmentation | None = None


def _get_model() -> SegformerForSemanticSegmentation:
    """
    Load the SegFormer model on first call; return the cached object on
    subsequent calls. CPU-only — no CUDA API is invoked.
    """
    global _model
    if _model is None:
        _model = SegformerForSemanticSegmentation.from_pretrained(
            MODEL_ID, 
            use_safetensors=True,
            low_cpu_mem_usage=True,
            dtype=torch.float16 # <--- Cambiado a 'dtype'
        )
        _model = _model.to("cpu")
        _model.eval()
    return _model


# ---------------------------------------------------------------------------
# Stage 0: Segmentation preprocessing (no transformers processor — s390x-safe)
# ---------------------------------------------------------------------------

def _preprocess_segformer(image_pil: Image.Image) -> dict:
    """
    Replicate `SegformerImageProcessor` behaviour with Pillow + NumPy + torch.
    """
    if image_pil.mode != "RGB":
        image_pil = image_pil.convert("RGB")

    image = image_pil.resize(
        (SEG_INPUT_SIZE, SEG_INPUT_SIZE), Image.Resampling.BILINEAR
    )

    arr = np.asarray(image, dtype=np.float32) / 255.0        # (512, 512, 3)
    arr = (arr - SEG_MEAN) / SEG_STD
    arr = np.transpose(arr, (2, 0, 1))                       # (3, 512, 512)

    # Convertimos explícitamente el tensor a half precision (float16) 
    # para que coincida con el modelo cargado y no cause conflicto de tipos
    pixel_values = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    pixel_values = pixel_values.unsqueeze(0).to(torch.float16) # (1, 3, 512, 512) en float16
    return {"pixel_values": pixel_values}


# ---------------------------------------------------------------------------
# Stage 1: Segmentation
# ---------------------------------------------------------------------------

def _run_segmentation(
    image_pil: Image.Image,
    model: SegformerForSemanticSegmentation,
    debug_dir: str | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Run SegFormer semantic segmentation on a PIL RGB image.

    Returns:
        mask_smoothed: uint8 numpy array, same H×W as input.
                       255 = inpaint this region, 0 = preserve.
        coverage:      dict with percentage coverage per semantic category.
                       {"water_body": float, "turf_ground": float,
                        "arboreal_canopy": float}

    If debug_dir is provided (or PIPELINE_DEBUG=1 is set), a labeled
    colour-overlay image is saved as debug_dir/debug_overlay.png.
    """
    w, h = image_pil.size

    inputs = _preprocess_segformer(image_pil)
    # Move inputs to CPU explicitly
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # shape: (1, num_classes, H', W')
    upsampled = F.interpolate(
        logits, size=(h, w), mode="bilinear", align_corners=False
    )
    probs = F.softmax(upsampled, dim=1)[0]          # (num_classes, H, W)
    pred_seg = upsampled.argmax(dim=1)[0].cpu().numpy()   # (H, W)
    max_probs = probs.max(dim=0).values.cpu().numpy()     # (H, W)
    total_pixels = pred_seg.size

    mask_binary = np.zeros(pred_seg.shape, dtype=np.uint8)
    coverage = {"water_body": 0.0, "turf_ground": 0.0, "arboreal_canopy": 0.0}

    unique_ids = np.unique(pred_seg)

    for cid in unique_ids:
        label = model.config.id2label.get(int(cid), f"ID_{cid}").lower()
        pixels = pred_seg == cid

        # Mask assignment: white = inpaint, black = preserve
        if any(kw in label for kw in IMPACT_KEYWORDS):
            mask_binary[pixels] = 255

        # Coverage accumulation into semantic buckets
        for category, keywords in COVERAGE_KEYWORDS.items():
            if any(kw in label for kw in keywords):
                coverage[category] += float(np.sum(pixels) / total_pixels * 100)
                break

    # Smooth mask edges with Gaussian blur
    mask_smoothed = _gaussian_blur_2d(mask_binary)

    # Debug overlay — produced when PIPELINE_DEBUG=1 or debug_dir is passed
    _write_debug_overlay(
        image_pil, pred_seg, model, unique_ids, debug_dir
    )

    return mask_smoothed, coverage


def _write_debug_overlay(
    image_pil: Image.Image,
    pred_seg: np.ndarray,
    model: SegformerForSemanticSegmentation,
    unique_ids: np.ndarray,
    debug_dir: str | None,
) -> None:
    """
    Write a colour-overlay debug image to debug_dir/debug_overlay.png.
    Only runs when PIPELINE_DEBUG=1 or debug_dir is explicitly provided.
    """
    if debug_dir is None and os.environ.get("PIPELINE_DEBUG") != "1":
        return

    import matplotlib.patheffects as pe  # deferred import — only needed for debug
    import matplotlib.pyplot as plt

    image_np = np.array(image_pil).astype(np.float32)
    h, w = image_np.shape[:2]
    cmap = plt.get_cmap("tab20")
    color_overlay = np.zeros((h, w, 3), dtype=np.uint8)

    for idx, cid in enumerate(unique_ids):
        label = model.config.id2label.get(int(cid), f"ID_{cid}")
        pixels = pred_seg == cid
        color_rgb = [
            int(c * 255)
            for c in cmap(idx / max(1, len(unique_ids) - 1))[:3]
        ]
        color_overlay[pixels] = color_rgb

    # Alpha blend, equivalent to cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
    blended = (image_np * 0.5 + color_overlay.astype(np.float32) * 0.5).clip(0, 255)
    blended = np.floor(blended + 0.5).astype(np.uint8)

    # Draw labels with matplotlib (Pillow's default font cannot scale).
    # Coordinates are identical: (cx, cy) in cv2 == (x, y) in matplotlib with origin='upper'.
    fig = plt.figure(figsize=(blended.shape[1] / 100, blended.shape[0] / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(blended, origin="upper")
    ax.set_axis_off()

    for cid in unique_ids:
        label = model.config.id2label.get(int(cid), f"ID_{cid}")
        pixels = pred_seg == cid
        y_coords, x_coords = np.where(pixels)
        if len(y_coords) > 0:
            cx = float(np.median(x_coords))
            cy = float(np.median(y_coords))
            ax.text(cx, cy, label, fontsize=7, ha="center", va="center",
                    color="white",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])

    out_path = os.path.join(debug_dir or ".", "debug_overlay.png")
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stage 2: Dynamic prompt assembly
# ---------------------------------------------------------------------------

def _build_prompt(coverage: dict, yaml_path: str) -> str:
    """
    Assemble the FLUX.1 Fill inpainting prompt from the YAML config
    and the coverage percentages produced by segmentation.
    """
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to load prompt config: {exc}") from exc

    parts = [config.get("base_header", "").strip()]
    classes = config.get("classes", {})

    for category, bands in THRESHOLDS.items():
        pct = coverage.get(category, 0.0)
        if pct <= 0:
            continue
        cat_config = classes.get(category, {})
        for lo, hi, band_key in bands:
            if lo < pct <= hi or (hi == 101 and pct > lo):
                text = cat_config.get(band_key, "")
                if text:
                    parts.append(text.strip())
                break

    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Stage 3: Image & mask preprocessing for FLUX.1 Fill
# ---------------------------------------------------------------------------

def _preprocess(
    image_pil: Image.Image,
    mask_pil: Image.Image,
    max_dim: int = MAX_DIM,
) -> tuple[Image.Image, Image.Image]:
    """
    Resize image and mask to meet FLUX.1 Fill dimension constraints:
    - Longest edge ≤ max_dim (default 1024px), aspect ratio preserved.
    - Both dimensions rounded down to nearest multiple of 16.

    Returns (image_resized, mask_resized) as PIL Images.
    """
    # Align mask to image size first
    if image_pil.size != mask_pil.size:
        mask_pil = mask_pil.resize(image_pil.size, Image.Resampling.NEAREST)

    w, h = image_pil.size

    # Scale down if necessary
    if max(w, h) > max_dim:
        ratio = max_dim / float(max(w, h))
        w = int(w * ratio)
        h = int(h * ratio)

    # Floor-round to 16px multiples
    new_w = (w // 16) * 16
    new_h = (h // 16) * 16

    # Ensure at least 16×16
    new_w = max(new_w, 16)
    new_h = max(new_h, 16)

    return (
        image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS),
        mask_pil.resize((new_w, new_h), Image.Resampling.LANCZOS),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(image_bytes: bytes, yaml_path: str) -> str:
    """
    Run the full image devastation pipeline.

    Args:
        image_bytes: Raw bytes of the input image (JPEG, PNG, WebP, etc.)
        yaml_path:   Path to the prompts YAML configuration file.

    Returns:
        A Replicate prediction URL (str). Poll this URL with a GET request
        to check job status and retrieve the output image when complete.

    Raises:
        ValueError:   REPLICATE_API_TOKEN is not set in the environment.
        RuntimeError: Any unrecoverable pipeline error (image decode failure,
                      YAML load failure, Replicate submission failure, etc.)
        OSError:      Temp directory creation failed (propagated as-is).
    """
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise ValueError("REPLICATE_API_TOKEN not set")

    with tempfile.TemporaryDirectory() as tmp:
        # --- 1. Decode input image ---
        try:
            image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to decode image: {exc}") from exc

        # --- 2. Segmentation ---
        model = _get_model()
        # Only write the debug overlay when explicitly requested; it is a hard
        # dependency on matplotlib, which is not always available (e.g. s390x).
        debug_dir = tmp if os.environ.get("PIPELINE_DEBUG") == "1" else None
        mask_np, coverage = _run_segmentation(
            image_pil, model, debug_dir=debug_dir
        )

        # --- 3. Dynamic prompt ---
        prompt = _build_prompt(coverage, yaml_path)

        # --- 4. Preprocessing ---
        mask_pil = Image.fromarray(mask_np, mode="L")
        img_ready, mask_ready = _preprocess(image_pil, mask_pil)

        # --- 5. Save preprocessed files to temp dir ---
        img_path = os.path.join(tmp, "img_ready.png")
        mask_path = os.path.join(tmp, "mask_ready.png")
        img_ready.save(img_path, format="PNG")
        mask_ready.save(mask_path, format="PNG")

        # --- 6. Submit to Replicate (non-blocking) ---
        try:
            client = replicate.Client(api_token=token)
            with open(img_path, "rb") as img_f, open(mask_path, "rb") as mask_f:
                prediction = client.predictions.create(
                    # Pass the model object, not the string. The string form
                    # leaves `version` unset and fails on replicate>=1.0.
                    model=client.models.get(REPLICATE_MODEL),
                    input={
                        "image": img_f,
                        "mask": mask_f,
                        "prompt": prompt,
                        "guidance": 35.0,
                        "num_inference_steps": 40,
                        "output_format": "png",
                        "output_quality": 100,
                        "megapixels": "1",
                    },
                )
        except Exception as exc:
            raise RuntimeError(f"Replicate submission failed: {exc}") from exc

        # --- 7. Return polling URL ---
        url = prediction.urls.get("get") if prediction.urls else None
        if not url:
            raise RuntimeError("Replicate prediction URL unavailable")

        return url
