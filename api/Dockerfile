# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.10-trixie

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin

# Copy dependency files first (for better layer caching)
COPY pyproject.toml uv.lock ./

# Install the project's dependencies
# Note: Using --no-install-project to install only dependencies first
RUN uv sync --no-install-project --no-dev

# Copy the rest of the project source code
COPY . /app

# Install the project itself
RUN uv sync --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

# Run the FastAPI application
# Uses `--host 0.0.0.0` to allow access from outside the container
CMD ["fastapi", "run", "--host", "0.0.0.0"]