# Base image already ships s390x builds of torch, numpy, pillow, transformers,
# replicate, pyyaml and python-dotenv. Do NOT `uv sync` / pip install the
# lockfile on top: it would try to replace torch with a PyPI build that has no
# s390x wheel.
FROM registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest

WORKDIR /app
COPY pipeline.py smoke_test.py prompts_generalized.yaml ./

# REPLICATE_API_TOKEN is injected at runtime via --env-file, never baked in.
# CMD is inherited from the base image (JupyterLab).
