#!/usr/bin/env bash
# ParadeDB lifecycle helper, run inside WSL2 Ubuntu.
#   bash db.sh up | down | reset | status | psql | ext
set -euo pipefail
DIR="/mnt/c/code/claude-memory/docker"
DC="sudo docker compose -f $DIR/docker-compose.yml"
CN="claudemem-db"

wait_healthy() {
  echo "--- waiting for healthy ---"
  for i in $(seq 1 40); do
    s="$(sudo docker inspect -f '{{.State.Health.Status}}' "$CN" 2>/dev/null || echo none)"
    echo "health=$s"
    [ "$s" = "healthy" ] && return 0
    sleep 3
  done
  echo "DB did not become healthy in time" >&2
  sudo docker logs --tail 40 "$CN" || true
  return 1
}

case "${1:-status}" in
  up)
    sudo systemctl start docker 2>/dev/null || true
    $DC up -d
    wait_healthy
    ;;
  down)    $DC down ;;
  reset)   $DC down -v && $DC up -d && wait_healthy ;;
  status)  sudo docker ps --filter "name=$CN" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' ;;
  psql)    shift; sudo docker exec -i "$CN" psql -U claudemem -d claudemem "$@" ;;
  ext)     sudo docker exec -i "$CN" psql -U claudemem -d claudemem -c "SELECT extname, extversion FROM pg_extension ORDER BY extname;" ;;
  *) echo "usage: db.sh up|down|reset|status|psql|ext" >&2; exit 1 ;;
esac
