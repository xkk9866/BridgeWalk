#!/usr/bin/env bash
# Usage: scp_r.sh <local files...> <remote subdirectory>
set -u
: "${SSH_HOST:?Set SSH_HOST}"
: "${SSH_USER:?Set SSH_USER}"
: "${SSH_REMOTE_DIR:?Set SSH_REMOTE_DIR}"
if [ "$#" -lt 2 ]; then
    echo "Usage: scp_r.sh <local files...> <remote subdirectory>" >&2
    exit 2
fi
destination="${SSH_USER}@${SSH_HOST}:${SSH_REMOTE_DIR%/}/${!#}/"
for attempt in $(seq 1 20); do
    timeout 300 scp -q -o BatchMode=yes -o ConnectTimeout=60 \
        -P "${SSH_PORT:-22}" "${@:1:$#-1}" "$destination" && exit 0
    sleep 15
done
exit 1
