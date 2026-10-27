#!/usr/bin/env bash
# Run an SSH command with bounded retries.
set -u
: "${SSH_HOST:?Set SSH_HOST}"
: "${SSH_USER:?Set SSH_USER}"
for attempt in 1 2 3 4 5; do
    timeout "${T:-600}" ssh -o BatchMode=yes -o ConnectTimeout=60 \
        -o ServerAliveInterval=30 -p "${SSH_PORT:-22}" \
        "${SSH_USER}@${SSH_HOST}" "$@" && exit 0
    status=$?
    [ "$status" -ne 255 ] && exit "$status"
    sleep 8
done
exit 255
