# Base image already ships s390x builds of torch, numpy, pillow, transformers,
# replicate, pyyaml and python-dotenv. Do NOT `uv sync` / pip install the
# lockfile on top: it would try to replace torch with a PyPI build that has no
# s390x wheel.
FROM registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest

# Everything the app imports except torch. Already-satisfied requirements are
# no-ops, so torch is never replaced. Versions match what the app was tested
# with (transformers 4.x, see pyproject.toml).
RUN pip install --no-cache-dir         fastapi uvicorn python-multipart         python-dotenv replicate pyyaml requests pillow numpy         "transformers>=4.57,<5"

# Fail the build (not the running container) if an import is missing.
RUN python -c "import torch, transformers, numpy, PIL, yaml, dotenv, replicate, requests, fastapi, uvicorn, multipart"

WORKDIR /app
COPY pipeline.py app.py smoke_test.py prompts_generalized.yaml ./
COPY dist ./dist

EXPOSE 8000

# REPLICATE_API_TOKEN is injected at runtime via --env-file, never baked in.
# This replaces the base image's JupyterLab as the container's main process.
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
