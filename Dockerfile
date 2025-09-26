# ------------ Stage 1: build ------------
FROM debian:bookworm AS builder
ARG DEBIAN_FRONTEND=noninteractive

# Python + build toolchain + native libs for SciPy, PyAudio, PyQt5
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    build-essential gcc g++ make pkg-config \
    libffi-dev \
    # Audio build deps
    libasound2-dev portaudio19-dev libpulse-dev \
    # Qt/OpenGL runtime (helps PyQt5 run, and lets PyInstaller find them)
    libgl1 libsm6 libxext6 libxrender1 libx11-6 \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create/enable venv and ensure it's on PATH so `pyinstaller` is resolvable
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Python deps (in the venv)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir pyinstaller numpy scipy pyqt5 pyqtgraph pyaudio pyyaml

# Build your app
WORKDIR /app
COPY . .
# Add flags as needed (e.g., --noconfirm --clean --strip)
RUN pyinstaller --onefile main.py

# ------------ Stage 2: runtime ------------
FROM debian:bookworm-slim AS runtime
ARG DEBIAN_FRONTEND=noninteractive

# Only the runtime libs your binary needs (audio + Qt basics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libasound2 libpulse0 libportaudio2 \
    libffi8 \
    libgl1 libsm6 libxext6 libxrender1 libx11-6 \
    # Helpful for many PyQt5 builds (XKB/XCB pieces)
    libxkbcommon0 libxkbcommon-x11-0 libxcb1 libxcb-xinerama0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled binary
COPY --from=builder /app/dist/main /usr/local/bin/app

# If you’re running headless containers with PyQt, uncomment:
# ENV QT_QPA_PLATFORM=offscreen

ENTRYPOINT ["/usr/local/bin/app"]
