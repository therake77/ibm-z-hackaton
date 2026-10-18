# Base image already ships s390x builds of torch, numpy, pillow, transformers,
# replicate, pyyaml and python-dotenv. Do NOT `uv sync` / pip install the
# lockfile on top: it would try to replace torch with a PyPI build that has no
# s390x wheel.
FROM registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest

# Web layer only (pure Python / existing s390x wheels). torch is untouched.
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

WORKDIR /app
COPY pipeline.py app.py smoke_test.py prompts_generalized.yaml ./
COPY dist ./dist

EXPOSE 8000

# REPLICATE_API_TOKEN is injected at runtime via --env-file, never baked in.
# This replaces the base image's JupyterLab as the container's main process.
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
