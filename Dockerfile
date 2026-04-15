FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dbmate for migrations
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && curl -fsSL -o /usr/local/bin/dbmate https://github.com/amacneil/dbmate/releases/latest/download/dbmate-linux-amd64 \
    && chmod +x /usr/local/bin/dbmate \
    && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock .
RUN uv sync --no-dev --no-install-project

COPY kombinat/ kombinat/
COPY db/ db/
COPY dbmate.toml .
COPY start.sh .
RUN chmod +x start.sh

RUN uv sync --no-dev

EXPOSE 8000

CMD ["./start.sh"]
