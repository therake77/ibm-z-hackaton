import base64
import io
import os
import platform
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
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


# ---------------------------------------------------------------------------
# Abuse controls. Every /api/process or /generate call can spend Replicate
# credits and several GB of RAM, so the public deployment is capped here.
# All limits can be overridden through environment variables.
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


MAX_UPLOAD_BYTES = _env_int("MAX_UPLOAD_MB", 8) * 1024 * 1024
JOB_RATE_LIMIT = _env_int("JOB_RATE_LIMIT", 3)                  # jobs per IP ...
JOB_RATE_WINDOW = _env_int("JOB_RATE_WINDOW_SECONDS", 600)      # ... per window
STATUS_RATE_LIMIT = _env_int("STATUS_RATE_LIMIT", 60)           # polls per IP ...
STATUS_RATE_WINDOW = 60                                         # ... per minute
MAX_JOBS_PER_DAY = _env_int("MAX_JOBS_PER_DAY", 50)             # Replicate jobs, all clients
JOB_TTL_SECONDS = 3600


class RateLimiter:
    """In-memory sliding-window limiter, keyed by client IP."""

    def __init__(self, limit: int, window: int):
        self.limit = limit
        self.window = window
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> int:
        """Record a hit. Returns 0 if allowed, else seconds until a slot frees."""
        now = time.monotonic()
        with self._lock:
            if len(self._hits) > 5000:  # bound memory under a flood of IPs
                for k in [k for k, q in self._hits.items()
                          if not q or now - q[-1] >= self.window]:
                    del self._hits[k]
            q = self._hits[key]
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                return max(1, int(self.window - (now - q[0])) + 1)
            q.append(now)
            return 0


_job_limiter = RateLimiter(JOB_RATE_LIMIT, JOB_RATE_WINDOW)
_status_limiter = RateLimiter(STATUS_RATE_LIMIT, STATUS_RATE_WINDOW)

# One pipeline run at a time: keeps memory bounded on a small VM.
_job_gate = threading.BoundedSemaphore(1)

_daily = {"day": None, "count": 0}
_daily_lock = threading.Lock()

# Prediction URLs this server handed out, so /api/status can't be used to
# query arbitrary predictions with our token.
_issued_jobs: dict[str, float] = {}
_issued_lock = threading.Lock()


def _client_ip(request: Request) -> str:
    # Behind the tunnel/proxy uvicorn rewrites this from X-Forwarded-For, but
    # only for the addresses listed in FORWARDED_ALLOW_IPS.
    return request.client.host if request.client else "unknown"


def _enforce_rate_limit(limiter: RateLimiter, request: Request) -> None:
    wait = limiter.check(_client_ip(request))
    if wait:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later.",
            headers={"Retry-After": str(wait)},
        )


def _daily_budget_left() -> bool:
    today = datetime.now(timezone.utc).date()
    with _daily_lock:
        if _daily["day"] != today:
            _daily.update(day=today, count=0)
        return _daily["count"] < MAX_JOBS_PER_DAY


def _record_replicate_job(job_url: str) -> None:
    now = time.monotonic()
    with _daily_lock:
        _daily["count"] += 1
    with _issued_lock:
        for url in [u for u, exp in _issued_jobs.items() if exp <= now]:
            del _issued_jobs[url]
        _issued_jobs[job_url] = now + JOB_TTL_SECONDS


def _is_issued_job(job_url: str) -> bool:
    with _issued_lock:
        exp = _issued_jobs.get(job_url)
        return exp is not None and exp > time.monotonic()


@contextmanager
def _job_slot():
    if not _job_gate.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="The server is busy with another image. Please retry in a minute.",
            headers={"Retry-After": "30"},
        )
    try:
        yield
    finally:
        _job_gate.release()


def _read_upload(file: UploadFile) -> bytes:
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)",
        )
    return data


def _safe_detail(exc: Exception) -> str:
    """Error text for clients; never let the API token through."""
    text = str(exc)
    token = os.environ.get("REPLICATE_API_TOKEN")
    return text.replace(token, "***") if token else text


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm SegFormer model on startup so the first request is fast
    try:
        pipeline._get_model()
    except Exception as exc:
        print(f"[WARN] Could not pre-warm SegFormer on startup: {exc}")
    yield


# No CORS middleware (the frontend is served from this same app) and no
# auto-generated /docs, /redoc or /openapi.json on a public deployment.
app = FastAPI(
    title="IBM Z Hackathon — SegFormer + FLUX.1 Fill Pipeline API",
    version="2.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # Cheap early rejection of oversized uploads (honest clients send the length).
    if request.method == "POST":
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_UPLOAD_BYTES + 64 * 1024:
            return JSONResponse({"detail": "Upload too large"}, status_code=413)
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


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
def generate(request: Request, file: UploadFile = File(...)):
    """Calls pipeline.run() and returns {"job_url": prediction_url}."""
    _enforce_rate_limit(_job_limiter, request)
    if not _daily_budget_left():
        raise HTTPException(
            status_code=429, detail="Daily generation limit reached. Try again tomorrow."
        )
    raw_bytes = _read_upload(file)
    with _job_slot():
        try:
            prediction_url = pipeline.run(raw_bytes, str(YAML_PATH))
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=_safe_detail(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=_safe_detail(exc)) from exc
        _record_replicate_job(prediction_url)
    return {"job_url": prediction_url}


@app.post("/api/process")
def process_image(
    request: Request,
    file: UploadFile = File(...),
    skip_replicate: bool = Form(False),
):
    """
    Interactive UI endpoint: runs the pipeline and returns the intermediate
    artifacts (preprocessed input, SegFormer mask, class overlay), semantic
    coverage percentages, the assembled prompt, and the Replicate job URL.
    With skip_replicate=true nothing is submitted to Replicate.
    """
    _enforce_rate_limit(_job_limiter, request)
    if not skip_replicate and not _daily_budget_left():
        raise HTTPException(
            status_code=429, detail="Daily generation limit reached. Try again tomorrow."
        )
    raw_bytes = _read_upload(file)
    with _job_slot():
        try:
            result = pipeline.run_detailed(
                raw_bytes,
                str(YAML_PATH),
                submit=not skip_replicate,
                with_overlay=True,
            )
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=_safe_detail(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=_safe_detail(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Pipeline error: {_safe_detail(exc)}"
            ) from exc
        if result["job_url"]:
            _record_replicate_job(result["job_url"])

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
    request: Request,
    job_url: str = Query(..., description="Replicate prediction polling URL"),
):
    """Poll the FLUX.1 Fill job status on Replicate."""
    _enforce_rate_limit(_status_limiter, request)
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="REPLICATE_API_TOKEN not set")
    if not _is_replicate_prediction_url(job_url):
        raise HTTPException(status_code=400, detail="Not a Replicate prediction URL")
    if not _is_issued_job(job_url):
        raise HTTPException(status_code=404, detail="Unknown or expired job")

    headers = {"Authorization": f"Token {token}"}
    try:
        resp = requests.get(job_url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Failed to query Replicate: {_safe_detail(exc)}"
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
    }


# Serve compiled frontend static bundle if present in ./dist
if STATIC_DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIST_DIR), html=True), name="static")
