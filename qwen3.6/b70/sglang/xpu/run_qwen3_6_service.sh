#!/bin/bash

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NAME="${NAME:-omarchy-local-ai}"
IMAGE="${IMAGE:-llm-scaler-sgl:bmg}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/models}"
HOST_PORT="${HOST_PORT:-8010}"

if docker container inspect "$NAME" >/dev/null 2>&1; then
  if [[ $(docker inspect --format '{{.State.Running}}' "$NAME") == "true" ]]; then
    echo "http://127.0.0.1:$HOST_PORT/v1"
  else
    docker start "$NAME" >/dev/null
    echo "http://127.0.0.1:$HOST_PORT/v1"
  fi
  exit
fi

[[ -d $MODEL_ROOT/Qwen3.6-35B-A3B ]] || { echo "Missing $MODEL_ROOT/Qwen3.6-35B-A3B" >&2; exit 1; }
docker image inspect "$IMAGE" >/dev/null

DRI_GROUP=$(stat -c %g /dev/dri/renderD128)
DRI_ARGS=(--device /dev/dri:/dev/dri --group-add "$DRI_GROUP")
[[ -d /dev/dri/by-path ]] && DRI_ARGS+=(-v /dev/dri/by-path:/dev/dri/by-path:ro)

docker run -d \
  --name "$NAME" \
  --restart unless-stopped \
  "${DRI_ARGS[@]}" \
  --shm-size 16g \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -p "127.0.0.1:$HOST_PORT:30000" \
  -v "$MODEL_ROOT:/models:rw" \
  -v "$ROOT/start_qwen3_6_service.sh:/start.sh:ro" \
  --entrypoint "" \
  "$IMAGE" \
  bash /start.sh >/dev/null

echo "http://127.0.0.1:$HOST_PORT/v1"
