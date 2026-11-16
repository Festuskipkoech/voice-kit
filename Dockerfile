# syntax=docker/dockerfile:1

# -- chef base: build deps + cargo-chef installed once, shared by all stages --
FROM rust:1.92-slim AS chef

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build libclang-dev libssl-dev pkg-config curl git \
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

# Deps compiled once; espeak-rs-sys runs CMake cleanly here and stays cached.
RUN cargo chef cook --release --recipe-path recipe.json

COPY . .
RUN cargo build --locked --release --bin voicekit-server

# -- runtime: minimal image, just the binary and its shared libs --
FROM debian:trixie-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libssl3 ca-certificates curl libespeak-ng1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --system --no-create-home --shell /usr/sbin/nologin voicekit

WORKDIR /app

COPY --from=builder /build/target/release/voicekit-server ./voicekit-server
COPY models.toml          /etc/voicekit/models.toml
COPY config.template.toml /etc/voicekit/config.template.toml

RUN mkdir -p /models /data && chown voicekit:voicekit /models /data

USER voicekit
EXPOSE 8080

ENTRYPOINT ["/app/voicekit-server"]
CMD ["--config", "/data/config.toml"]