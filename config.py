# -*- coding: utf-8 -*-
# Конфигурация и загрузка параметров из .env.
from dataclasses import dataclass
from typing import Dict, List, Optional
import os

from dotenv import load_dotenv
import yaml


@dataclass
class Config:
    MODE: str
    MARKETS: List[str]
    VERBOSE: bool
    VERBOSE_CTX: bool
    SLEEP_SEC: int
    EXCHANGE: str
    API_KEY: Optional[str]
    API_SECRET: Optional[str]
    CCXT_TIMEOUT_MS: int = 30000
    CCXT_RETRY_COUNT: int = 3
    CCXT_RETRY_BACKOFF_SEC: float = 1.0
    STRATEGY: str = "first"


@dataclass
class StrategyConfig:
    NAME: str = "first"
    TIMEFRAME: str = "1m"
    LOOKBACK: int = 600
    RISK_PCT: float = 0.005
    ATR_K: float = 3.0
    ATR_PCT_MIN: float = 0.002
    ATR_PCT_MAX: float = 0.08
    MAX_DAILY_DD_PCT: float = 0.02
    MAX_LOSSES_IN_ROW: int = 3
    FAST_MODE: bool = False
    FAST_HTF: str = "5m"
    FAST_RSI_MIN: int = 60
    FAST_ATR_K: float = 3.0
    FAST_ATR_PCT_MIN: float = 0.002
    FAST_ATR_PCT_MAX: float = 0.08
    FAST_MIN_VOL_SMA: int = 20
    TP_R_MULT: float = 2.0
    USE_TP: bool = True
    USE_OCO_WHEN_AVAILABLE: bool = True
    FIXED_STOP_EUR: Optional[float] = 0.0
    STOP_MAX_PCT: float = 0.3
    TARGET_ENTRY_COST: float = 10.0
    HARD_STOP_LOSS_PCT: float = 0.15
    ENABLE_EOD_EXIT: bool = True
    EOD_EXIT_HOUR: int = 23
    STRUCTURE_LOOKBACK: int = 6
    STRUCTURE_BUFFER_ATR_K: float = 0.5
    MIN_STOP_ATR_K: float = 2.0
    MIN_VOL_MULT: float = 1.0
    MIN_BODY_PCT: float = 0.0
    COOLDOWN_BARS: int = 0
    MAX_TRADES_PER_DAY: int = 0
    MIN_RR: float = 2.0
    ENABLE_TRAIL_PROFIT: bool = False
    TRAIL_PROFIT_TRIGGER_PCT: float = 0.02
    TRAIL_PROFIT_OFFSET_PCT: float = 0.01
    TRAIL_PROFIT_MIN_LOCK_PCT: float = 0.001
    TRAIL_PROFIT_MIN_MOVE_PCT: float = 0.002
    TRAIL_PROFIT_STOP_LIMIT_GAP_PCT: float = 0.001


def parse_bool(v: str, default=False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in ('1', 'true', 'yes', 'y', 'on')


def load_config() -> Config:
    load_dotenv()
    mode = os.getenv('MODE', 'paper').strip().lower()
    markets_env = os.getenv('MARKETS', 'BTC/EUR').replace(' ', '')
    markets = [m for m in markets_env.split(',') if m]
    cfg = Config(
        MODE=mode,
        MARKETS=markets,
        VERBOSE=parse_bool(os.getenv('VERBOSE', 'true'), True),
        VERBOSE_CTX=parse_bool(os.getenv('VERBOSE_CTX', 'false'), False),
        SLEEP_SEC=int(os.getenv('SLEEP_SEC', '3')),
        EXCHANGE=os.getenv('EXCHANGE', 'binance').lower(),
        API_KEY=os.getenv('API_KEY') or None,
        API_SECRET=os.getenv('API_SECRET') or os.getenv('API_SECRET_KEY') or None,
        CCXT_TIMEOUT_MS=int(os.getenv('CCXT_TIMEOUT_MS', '30000')),
        CCXT_RETRY_COUNT=int(os.getenv('CCXT_RETRY_COUNT', '3')),
        CCXT_RETRY_BACKOFF_SEC=float(os.getenv('CCXT_RETRY_BACKOFF_SEC', '1.0')),
        STRATEGY=os.getenv('STRATEGY', 'first').strip().lower(),
    )
    return cfg


def _normalize_yaml_keys(data: Dict) -> Dict:
    out: Dict = {}
    for k, v in (data or {}).items():
        if k is None:
            continue
        key = str(k).strip().replace('-', '_').upper()
        out[key] = v
    return out


def load_strategy_config(name: str) -> StrategyConfig:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "strategies", f"{name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"strategy config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    data = _normalize_yaml_keys(raw)

    cfg = StrategyConfig()
    for field in cfg.__dataclass_fields__:
        if field in data:
            setattr(cfg, field, data[field])
    if getattr(cfg, "NAME", None) is None:
        cfg.NAME = name
    return cfg
