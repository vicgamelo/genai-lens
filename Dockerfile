FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --extra server --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --extra server

FROM python:3.12-slim-bookworm
RUN useradd --create-home --uid 10001 app && mkdir /data && chown app /data
COPY --from=build --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH" PORT=8080 GENAI_LENS_DATABASE_URL=sqlite:////data/genai-lens.db
USER app
VOLUME /data
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/health')"
CMD ["genai-lens-server"]
