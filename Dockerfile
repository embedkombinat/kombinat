FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock .
RUN uv sync --no-dev --no-install-project

COPY kombinat/ kombinat/

RUN uv sync --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "kombinat.main:app", "--host", "0.0.0.0", "--port", "8000"]
