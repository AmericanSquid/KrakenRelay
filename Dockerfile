# syntax=docker/dockerfile:1

# Choose your Debian suite:
# - bookworm = Debian 12
# - trixie   = Debian 13 (or equivalent testing/stable depending on date)
ARG DEBIAN_SUITE=bookworm

# -------------------------
# Build stage (compile deps)
# -------------------------
FROM debian:${DEBIAN_SUITE}-slim AS build

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps for:
# - building python packages (build-essential, gcc)
# - PyAudio (portaudio dev headers)
# - ALSA headers (libasound2-dev)
# - CFFI extension build (libffi-dev)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    build-essential gcc pkg-config \
    portaudio19-dev libasound2-dev libffi-dev \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Create venv (avoids Debian pip "externally managed" issues)
ENV VENV_PATH=/opt/venv
RUN python3 -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

# Always upgrade pip tooling (your preference)
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy dependency manifest first for caching
COPY requirements.txt /app/requirements.txt

# Optional: strip requirements that are either stdlib or heavy/unneeded
# - 'logging' is stdlib (pip package not needed)
# - SciPy is large and often unnecessary in your current KrakenRelay pipeline
# If you actually need SciPy, delete the grep and install requirements.txt directly.
RUN grep -vE '^(logging==|scipy==)' /app/requirements.txt > /tmp/requirements.runtime.txt

RUN python -m pip install --no-cache-dir -r /tmp/requirements.runtime.txt

# Copy the rest of the repo
COPY . /app

# Build the CFFI DSP module
# (Assumes kraken_dsp/build_dsp.py exists and produces kraken_dsp/_kraken_dsp*.so)
WORKDIR /app/kraken_dsp
RUN python build_dsp.py

WORKDIR /app


# -------------------------
# Runtime stage (small-ish)
# -------------------------
FROM debian:${DEBIAN_SUITE}-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Runtime libs:
# - libportaudio2 for PyAudio
# - libasound2 for ALSA
# - libffi8 (Debian 12) / libffi* for CFFI linkage
# - tzdata/ca-certificates are nice to have
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libportaudio2 \
    libasound2 \
    libffi8 \
    tzdata \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Copy venv + app from build stage
ENV VENV_PATH=/opt/venv
ENV PATH="${VENV_PATH}/bin:${PATH}"
COPY --from=build /opt/venv /opt/venv
COPY --from=build /app /app

# Helpful runtime env
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# If your Flask UI listens elsewhere, change these
EXPOSE 5000

# Run KrakenRelay (adjust args if your main.py expects them)
CMD ["python", "-u", "main.py"]
