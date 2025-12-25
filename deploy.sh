#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="stanislav@89.22.103.173"
REMOTE_DIR="/var/www/vhosts/inversi.org/trading_2026"
BRANCH="main"

VENV_DIR="${REMOTE_DIR}/.venv"
PY_BIN="${VENV_DIR}/bin/python"
PIP_BIN="${VENV_DIR}/bin/pip"

APP_ENTRY="bot.py"
LOG_FILE="bot.log"
PID_FILE="bot.pid"
REQ_FILE="requirements.txt"
REQ_HASH_FILE=".requirements.sha256"

echo "==> Push to GitHub (${BRANCH})"
git push origin "${BRANCH}"

echo "==> Deploy on server"
ssh "${REMOTE_HOST}" \
  "REMOTE_DIR='${REMOTE_DIR}' BRANCH='${BRANCH}' VENV_DIR='${VENV_DIR}' \
PY_BIN='${PY_BIN}' PIP_BIN='${PIP_BIN}' APP_ENTRY='${APP_ENTRY}' \
LOG_FILE='${LOG_FILE}' PID_FILE='${PID_FILE}' REQ_FILE='${REQ_FILE}' \
REQ_HASH_FILE='${REQ_HASH_FILE}' bash -s" <<'REMOTE'
set -euo pipefail
cd "${REMOTE_DIR}"

echo "==> git pull"
git fetch origin "${BRANCH}"
git checkout "${BRANCH}"
git pull --ff-only origin "${BRANCH}"

echo "==> ensure venv"
if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi

echo "==> install deps if requirements changed"
REQ_HASH=$(shasum -a 256 "${REQ_FILE}" | awk '{print $1}')
PREV_HASH=""
if [ -f "${REQ_HASH_FILE}" ]; then
  PREV_HASH=$(cat "${REQ_HASH_FILE}")
fi
if [ "${REQ_HASH}" != "${PREV_HASH}" ]; then
  PIP_LOG=$(mktemp)
  if ! "${PIP_BIN}" install -r "${REQ_FILE}" >"${PIP_LOG}" 2>&1; then
    echo "Requirements - failed"
    tail -n 50 "${PIP_LOG}" || true
    rm -f "${PIP_LOG}"
    exit 1
  fi
  rm -f "${PIP_LOG}"
  echo "${REQ_HASH}" > "${REQ_HASH_FILE}"
fi
echo "Requirements - ok"

echo "==> stop bot (if running)"
if [ -f "${PID_FILE}" ]; then
  PID=$(cat "${PID_FILE}")
  if kill -0 "${PID}" >/dev/null 2>&1; then
    kill "${PID}"
    sleep 1
  fi
  rm -f "${PID_FILE}"
fi
pkill -f "${APP_ENTRY}" >/dev/null 2>&1 || true

echo "==> start bot"
nohup "${PY_BIN}" "${APP_ENTRY}" > "${LOG_FILE}" 2>&1 &
echo $! > "${PID_FILE}"

echo "==> status check"
sleep 2
PID=$(cat "${PID_FILE}")
if kill -0 "${PID}" >/dev/null 2>&1; then
  echo "OK: бот запущен, pid=${PID}"
  echo "--- tail ${LOG_FILE} ---"
  tail -n 20 "${LOG_FILE}" || true
else
  echo "FAIL: процесс не запущен"
  echo "--- tail ${LOG_FILE} ---"
  tail -n 50 "${LOG_FILE}" || true
  exit 1
fi

echo "==> done"
REMOTE
