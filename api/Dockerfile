# OpenCGM-StateEvent API server.
#
# A minimal image for Hugging Face Spaces, Render, Fly.io, Railway, or any plain Docker
# host. The model weights are mounted at runtime — never bake them into the image.
#
# Build:
#     docker build -t opencgm-stateevent-api -f api/Dockerfile .
#
# Run (mount the released checkpoint + heads):
#     docker run --rm -p 8000:8000 \
#         -v /path/to/runs_5090:/app/runs_5090:ro \
#         -v /path/to/artifacts:/app/artifacts:ro \
#         -e OPENCGM_CHECKPOINT=/app/runs_5090/rawstats120/ckpt_ep040.pt \
#         -e OPENCGM_HEADS=/app/artifacts/heads.pkl \
#         opencgm-stateevent-api

FROM python:3.12-slim

WORKDIR /app

# Layer-cache friendly: copy only the manifest first, install deps, then copy sources.
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY api ./api

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir uv \
 && uv sync --frozen --no-dev \
 && uv pip install "fastapi>=0.110" "uvicorn[standard]>=0.27"

# Defaults match the released artefact paths. Override with -e at run time.
ENV OPENCGM_CHECKPOINT=/app/runs_5090/rawstats120/ckpt_ep040.pt \
    OPENCGM_HEADS=/app/artifacts/heads.pkl

EXPOSE 8000

# Single uvicorn worker is correct here: the model is loaded once at startup and the
# request handler is CPU-bound. Scale by running more containers behind a load balancer,
# not by adding workers inside one container (which would duplicate the model in memory).
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
