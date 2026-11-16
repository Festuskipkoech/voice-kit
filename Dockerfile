# syntax=docker/dockerfile:1

# -- chef base: build deps + cargo-chef installed once, shared by all stages --
FROM rust:1.92-slim AS chef

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build libclang-dev libssl-dev pkg-config curl git \
        libespeak-ng-dev \
    && rm -rf /var/lib/apt/lists/*

RUN cargo install --locked cargo-chef \
    && rustup component add rustfmt

WORKDIR /build

# -- planner: produce recipe.json from Cargo manifests, no source needed --
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# -- builder: cook deps (cached layer), then compile the real binary --
FROM chef AS builder

COPY --from=planner /build/recipe.json recipe.json

RUN cargo chef cook --release --recipe-path recipe.json

COPY . .
RUN cargo build --locked --release --bin voicekit-server

# Mirror espeak-ng-data into EVERY espeak-rs-sys-* build output directory.
#
# cargo-chef cook and cargo build can produce different hash-suffixed output
# dirs for espeak-rs-sys; the binary is baked against whichever dir the final
# `cargo build` used.  Rather than guessing which dir is the right one, we
# populate out/share/espeak-ng-data in every espeak-rs-sys-* dir found and
# mirror them all.  The runtime COPY replays the whole set so the binary finds
# its data regardless of which hash was baked in.
RUN data=$(find /build/target/release/build -type d -name espeak-ng-data 2>/dev/null \
        | head -1) && \
    [ -n "$data" ] || { echo "ERROR: no espeak-ng-data in build tree"; exit 1; } && \
    find /build/target/release/build -maxdepth 3 \
        -name 'espeak-rs-sys-*' -type d 2>/dev/null \
        > /tmp/_espeak_dirs.txt && \
    [ -s /tmp/_espeak_dirs.txt ] || { \
        echo "ERROR: no espeak-rs-sys-* dirs found"; \
        ls /build/target/release/build/ | head -30; exit 1; } && \
    mkdir -p /espeak-mirror && \
    while IFS= read -r pkg_dir; do \
        share="$pkg_dir/out/share/espeak-ng-data"; \
        printf 'mirror: %s\n' "$pkg_dir"; \
        mkdir -p "$(dirname "$share")"; \
        cp -r "$data/." "$share/"; \
        cp -r "$pkg_dir" /espeak-mirror/; \
    done < /tmp/_espeak_dirs.txt

# -- runtime: minimal image, binary + scripts + config baked in --
FROM debian:trixie-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libssl3 ca-certificates curl libespeak-ng1 python3 python3-pip \
    && pip install --no-cache-dir --break-system-packages "huggingface_hub>=1.0" tomli \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --system --no-create-home --shell /usr/sbin/nologin voicekit

WORKDIR /app

COPY --from=builder /build/target/release/voicekit-server ./voicekit-server

# Recreate the exact espeak-ng-data path the binary was compiled against.
# /espeak-mirror/ contains one directory: espeak-rs-sys-<hash>/
# Copying its contents into /build/target/release/build/ recreates
# the full path the binary expects (~4 MB).
COPY --from=builder /espeak-mirror/ /build/target/release/build/

# Bake all operator-facing files into the image
COPY models.toml                /etc/voicekit/models.toml
COPY config.template.toml       /etc/voicekit/config.template.toml
COPY scripts/init_config.sh     /etc/voicekit/init_config.sh
COPY scripts/download_models.py /etc/voicekit/download_models.py

RUN chmod +x /etc/voicekit/init_config.sh \
    && mkdir -p /models /data \
    && chown voicekit:voicekit /models /data

USER voicekit
EXPOSE 8080

ENTRYPOINT ["/app/voicekit-server"]
CMD ["--config", "/data/config.toml"]