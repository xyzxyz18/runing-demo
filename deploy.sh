#!/usr/bin/env bash
# Clone the repository, then run ./deploy.sh on a Linux server.
set -Eeuo pipefail

cd "$(dirname "$0")"
# Optional local overrides.  Copy .env.example to .env once; the defaults
# below keep a plain `./deploy.sh` fully usable without any extra setup.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
PORT="${PORT:-8000}"
BIND_ADDRESS="${BIND_ADDRESS:-0.0.0.0}"
PACE_MAX_UPLOAD_GB="${PACE_MAX_UPLOAD_GB:-2}"
if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "PORT 必须是 1-65535 之间的整数" >&2
  exit 1
fi
export PORT BIND_ADDRESS
export PACE_MAX_UPLOAD_GB

as_root() {
  if (( EUID == 0 )); then "$@"; else sudo "$@"; fi
}

install_docker_if_needed() {
  local need_docker=0 need_compose=0
  command -v docker >/dev/null 2>&1 || need_docker=1
  if (( need_docker == 1 )) || { ! docker compose version >/dev/null 2>&1 && ! command -v docker-compose >/dev/null 2>&1; }; then
    need_compose=1
  fi
  if (( need_docker == 0 && need_compose == 0 )); then return; fi
  if [[ "$(uname -s)" != Linux ]] || ! command -v apt-get >/dev/null 2>&1; then
    echo "未找到 Docker 或 Compose。请先安装 Docker Engine 和 Compose，再重新运行 ./deploy.sh。" >&2
    exit 1
  fi
  echo "正在安装缺失的 Docker 或 Compose…"
  as_root apt-get update
  if (( need_docker == 1 )); then as_root apt-get install -y docker.io; fi
  if (( need_compose == 0 )); then return; fi
  local compose_package=""
  for candidate in docker-compose-plugin docker-compose-v2 docker-compose; do
    if apt-cache show "$candidate" >/dev/null 2>&1; then
      compose_package="$candidate"
      break
    fi
  done
  if [[ -z "$compose_package" ]]; then
    echo "当前系统软件源没有 Compose，请安装 Docker Compose 后重试。" >&2
    exit 1
  fi
  as_root apt-get install -y "$compose_package"
}

install_docker_if_needed
if [[ "$(uname -s)" == Linux ]] && command -v systemctl >/dev/null 2>&1; then
  if ! docker info >/dev/null 2>&1; then as_root systemctl enable --now docker; fi
fi
DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
  if sudo -n docker info >/dev/null 2>&1 || { [[ -t 0 ]] && sudo docker info >/dev/null 2>&1; }; then
    DOCKER=(sudo docker)
  else
    echo "Docker 服务未启动或当前用户没有权限访问。请启动 Docker 后重试。" >&2
    exit 1
  fi
fi

compose() {
  if "${DOCKER[@]}" compose version >/dev/null 2>&1; then
    "${DOCKER[@]}" compose -f compose.yaml "$@"
  elif command -v docker-compose >/dev/null 2>&1; then
    if [[ "${DOCKER[0]}" == sudo ]]; then sudo docker-compose -f compose.yaml "$@"; else docker-compose -f compose.yaml "$@"; fi
  else
    echo "未找到 Docker Compose。请安装 Compose 后重试。" >&2
    exit 1
  fi
}

compose config --quiet
printf '正在构建并启动跑姿分析服务…\n'
compose up -d --build --remove-orphans
container_id="$(compose ps -q pace-lab)"
if [[ -z "$container_id" ]]; then
  echo "服务容器未创建，请运行 docker compose logs pace-lab 查看原因。" >&2
  exit 1
fi
for attempt in $(seq 1 90); do
  health="$("${DOCKER[@]}" inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container_id")"
  if [[ "$health" == healthy ]]; then
    if [[ "$BIND_ADDRESS" == 0.0.0.0 ]]; then
      printf '\n部署成功： http://<服务器公网 IP>:%s\n' "$PORT"
    else
      printf '\n部署成功： http://%s:%s\n' "$BIND_ADDRESS" "$PORT"
    fi
    printf '本机验证： http://127.0.0.1:%s/healthz\n' "$PORT"
    printf '如外网无法访问，请在服务器防火墙和云平台安全组开放 TCP %s。\n' "$PORT"
    printf '查看日志： docker compose logs -f pace-lab\n'
    exit 0
  fi
  if [[ "$health" == unhealthy || "$health" == exited ]]; then break; fi
  sleep 2
done
echo "服务未通过健康检查，最近日志：" >&2
compose logs --tail=80 pace-lab >&2
exit 1
