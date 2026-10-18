import base64
import io
import os
import platform
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from replicate.prediction import Prediction as _ReplicatePrediction

# Compatibility fix: allow version=None in Replicate Prediction model
# (official models return no version). Best effort: depends on the pydantic
# major version replicate was installed with.
try:
    if "version" in getattr(_ReplicatePrediction, "__fields__", {}):
        _ReplicatePrediction.__fields__["version"].allow_none = True
        _ReplicatePrediction.__fields__["version"].required = False
except Exception as exc:
    print(f"[WARN] Could not patch Replicate Prediction.version: {exc}")

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

import pipeline  # noqa: E402

YAML_PATH = BASE_DIR / "prompts_generalized.yaml"
STATIC_DIST_DIR = BASE_DIR / "dist"

REPLICATE_API_HOST = "api.replicate.com"
REPLICATE_PREDICTIONS_PREFIX = "/v1/predictions/"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm SegFormer model on startup so the first request is fast
    try:
        pipeline._get_model()
    except Exception as exc:
        print(f"[WARN] Could not pre-warm SegFormer on startup: {exc}")
    yield


app = FastAPI(
    title="IBM Z Hackathon — SegFormer + FLUX.1 Fill Pipeline API",
    version="2.0.0",
    lifespan=lifespan,
)

# The frontend is served from this same app, so credentials are not needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def encode_pil_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _is_replicate_prediction_url(url: str) -> bool:
    """The status endpoint sends our API token, so only talk to Replicate."""
    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.hostname == REPLICATE_API_HOST
        and parsed.path.startswith(REPLICATE_PREDICTIONS_PREFIX)
    )


# The handlers below are plain `def` on purpose: segmentation is CPU-bound and
# blocking, so FastAPI runs them in its threadpool instead of the event loop.

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "pipeline": "ready",
        "architecture": f"{platform.system()} ({platform.machine()})",
        "segmentation_model": pipeline.MODEL_ID,
        "inpainting_model": pipeline.REPLICATE_MODEL,
        "replicate_token_configured": bool(os.environ.get("REPLICATE_API_TOKEN")),
        "model_loaded": pipeline._model is not None,
    }


@app.post("/generate")
def generate(file: UploadFile = File(...)):
    """Calls pipeline.run() and returns {"job_url": prediction_url}."""
    raw_bytes = file.file.read()
    try:
        prediction_url = pipeline.run(raw_bytes, str(YAML_PATH))
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"job_url": prediction_url}


@app.post("/api/process")
def process_image(
    file: UploadFile = File(...),
    skip_replicate: bool = Form(False),
):
    """
    Interactive UI endpoint: runs the pipeline and returns the intermediate
    artifacts (preprocessed input, SegFormer mask, class overlay), semantic
    coverage percentages, the assembled prompt, and the Replicate job URL.
    With skip_replicate=true nothing is submitted to Replicate.
    """
    raw_bytes = file.file.read()
    try:
        result = pipeline.run_detailed(
            raw_bytes,
            str(YAML_PATH),
            submit=not skip_replicate,
            with_overlay=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Pipeline error: {exc}") from exc

    img_ready: Image.Image = result["img_ready"]
    mask_ready: Image.Image = result["mask_ready"]
    overlay = result["overlay"]
    orig_size = result["orig_size"]
    norm_size = img_ready.size
    coverage = {k: round(float(v), 2) for k, v in result["coverage"].items()}

    return {
        "status": "completed_local" if skip_replicate else "submitted",
        "job_url": result["job_url"],
        "original_size": {"width": orig_size[0], "height": orig_size[1]},
        "normalized_size": {"width": norm_size[0], "height": norm_size[1]},
        "area_proportions": coverage,
        "dynamic_prompt": result["prompt"],
        "images": {
            "input": encode_pil_base64(img_ready),
            "mask": encode_pil_base64(mask_ready),
            "overlay": encode_pil_base64(overlay) if overlay is not None else None,
            "result": None,
        },
    }


@app.get("/status")
@app.get("/api/status")
def check_prediction_status(
    job_url: str = Query(..., description="Replicate prediction polling URL"),
):
    """Poll the FLUX.1 Fill job status on Replicate."""
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="REPLICATE_API_TOKEN not set")
    if not _is_replicate_prediction_url(job_url):
        raise HTTPException(status_code=400, detail="Not a Replicate prediction URL")

    headers = {"Authorization": f"Token {token}"}
    try:
        resp = requests.get(job_url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Failed to query Replicate: {exc}"
        ) from exc

    status = data.get("status")
    raw_output = data.get("output")
    output_url = None
    if status == "succeeded" and raw_output:
        if isinstance(raw_output, list) and len(raw_output) > 0:
            output_url = raw_output[0]
        elif isinstance(raw_output, str):
            output_url = raw_output

    return {
        "status": status,
        "output_url": output_url,
        "error": data.get("error"),
        "logs": data.get("logs"),
    }


# Serve compiled frontend static bundle if present in ./dist
if STATIC_DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIST_DIR), html=True), name="static")
