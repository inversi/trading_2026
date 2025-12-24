# -*- coding: utf-8 -*-
# Утилиты логирования и форматирования.
import logging
import os
import pathlib
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

LOGGER_APP = None
LOGGER_TRADES = None
LOGGER_ERRORS = None


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')


def setup_logging(base_dir: str = 'logs'):
    global LOGGER_APP, LOGGER_TRADES, LOGGER_ERRORS
    log_dir = pathlib.Path(base_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = int(os.getenv("LOG_MAX_BYTES", 9 * 1024 * 1024))
    backup_count = int(os.getenv("LOG_BACKUP_COUNT", 30))

    formatter = logging.Formatter('[%(asctime)s UTC] %(message)s')
    formatter.converter = time.gmtime

    def make_handler(filename: str, level: int) -> RotatingFileHandler:
        h = RotatingFileHandler(
            log_dir / filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        h.setLevel(level)
        h.setFormatter(formatter)
        return h

    LOGGER_APP = logging.getLogger('app')
    LOGGER_APP.setLevel(logging.INFO)
    LOGGER_APP.handlers.clear()
    LOGGER_APP.addHandler(make_handler('app.log', logging.INFO))

    LOGGER_TRADES = logging.getLogger('trades')
    LOGGER_TRADES.setLevel(logging.INFO)
    LOGGER_TRADES.handlers.clear()
    LOGGER_TRADES.addHandler(make_handler('trades.log', logging.INFO))

    LOGGER_ERRORS = logging.getLogger('errors')
    LOGGER_ERRORS.setLevel(logging.ERROR)
    LOGGER_ERRORS.handlers.clear()
    LOGGER_ERRORS.addHandler(make_handler('errors.log', logging.ERROR))

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    LOGGER_APP.addHandler(console)
    LOGGER_TRADES.addHandler(console)
    LOGGER_ERRORS.addHandler(console)


def log_info(msg: str):
    if LOGGER_APP is not None:
        LOGGER_APP.info(msg)
    else:
        print(f"[{now_ts()}] {msg}")


def log_trade(msg: str):
    if LOGGER_TRADES is not None:
        LOGGER_TRADES.info(msg)
    else:
        print(f"[{now_ts()}] {msg}")


def log_error(msg: str, exc: Exception = None):
    if LOGGER_ERRORS is not None:
        if exc is not None:
            LOGGER_ERRORS.error(msg + f" | исключение={exc}", exc_info=True)
        else:
            LOGGER_ERRORS.error(msg)
    else:
        print(f"[{now_ts()}] ОШИБКА: {msg}")


def log(msg: str, verbose=True):
    if verbose:
        log_info(msg)


def fmt_float(x: float, digits: int = 8) -> str:
    try:
        return ("{0:." + str(digits) + "f}").format(float(x))
    except Exception:
        return str(x)


def format_ctx(ctx: dict, digits: int = 8) -> dict:
    out = {}
    for k, v in ctx.items():
        if isinstance(v, float):
            out[k] = fmt_float(v, digits)
        else:
            out[k] = v
    return out


def format_ctx_str(ctx: dict, digits: int = 8) -> str:
    f = format_ctx(ctx, digits)
    parts = []
    for k, v in f.items():
        parts.append(f"{k}={v}")
    return "{" + ", ".join(parts) + "}"
