# -*- coding: utf-8 -*-
# Конфигурация и загрузка параметров из .env.
from dataclasses import dataclass
from typing import List, Optional
import os

from dotenv import load_dotenv


@dataclass
class Config:
    MODE: str
    MARKETS: List[str]
    TIMEFRAME: str
    LOOKBACK: int
    RISK_PCT: float
    ATR_K: float
    ATR_PCT_MIN: float
    ATR_PCT_MAX: float
    MAX_DAILY_DD_PCT: float
    MAX_LOSSES_IN_ROW: int
    VERBOSE: bool
    VERBOSE_CTX: bool
    SLEEP_SEC: int
    FAST_MODE: bool
    FAST_HTF: str
    FAST_RSI_MIN: int
    FAST_ATR_K: float
    FAST_ATR_PCT_MIN: float
    FAST_ATR_PCT_MAX: float
    FAST_MIN_VOL_SMA: int
    EXCHANGE: str
    API_KEY: Optional[str]
    API_SECRET: Optional[str]
    TP_R_MULT: float
    USE_TP: bool
    USE_OCO_WHEN_AVAILABLE: bool
    FIXED_STOP_EUR: Optional[float] = None
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
    CCXT_TIMEOUT_MS: int = 30000
    CCXT_RETRY_COUNT: int = 3
    CCXT_RETRY_BACKOFF_SEC: float = 1.0
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
        TIMEFRAME=os.getenv('TIMEFRAME', '1m'),
        LOOKBACK=int(os.getenv('LOOKBACK', '600')),
        RISK_PCT=float(os.getenv('RISK_PCT', '0.005')),
        ATR_K=float(os.getenv('ATR_K', '3.0')),
        ATR_PCT_MIN=float(os.getenv('ATR_PCT_MIN', '0.002')),
        ATR_PCT_MAX=float(os.getenv('ATR_PCT_MAX', '0.08')),
        MAX_DAILY_DD_PCT=float(os.getenv('MAX_DAILY_DD_PCT', '0.02')),
        MAX_LOSSES_IN_ROW=int(os.getenv('MAX_LOSSES_IN_ROW', '3')),
        VERBOSE=parse_bool(os.getenv('VERBOSE', 'true'), True),
        VERBOSE_CTX=parse_bool(os.getenv('VERBOSE_CTX', 'false'), False),
        SLEEP_SEC=int(os.getenv('SLEEP_SEC', '3')),
        FAST_MODE=parse_bool(os.getenv('FAST_MODE', 'false'), False),
        FAST_HTF=os.getenv('FAST_HTF', '5m'),
        FAST_RSI_MIN=int(os.getenv('FAST_RSI_MIN', '60')),
        FAST_ATR_K=float(os.getenv('FAST_ATR_K', '3.0')),
        FAST_ATR_PCT_MIN=float(os.getenv('FAST_ATR_PCT_MIN', '0.002')),
        FAST_ATR_PCT_MAX=float(os.getenv('FAST_ATR_PCT_MAX', '0.08')),
        FAST_MIN_VOL_SMA=int(os.getenv('FAST_MIN_VOL_SMA', '20')),
        EXCHANGE=os.getenv('EXCHANGE', 'binance').lower(),
        API_KEY=os.getenv('API_KEY') or None,
        API_SECRET=os.getenv('API_SECRET') or os.getenv('API_SECRET_KEY') or None,
        TP_R_MULT=float(os.getenv('TP_R_MULT', '2.0')),
        USE_TP=parse_bool(os.getenv('USE_TP', 'true'), True),
        USE_OCO_WHEN_AVAILABLE=parse_bool(os.getenv('USE_OCO_WHEN_AVAILABLE', 'true'), True),
        FIXED_STOP_EUR=float(os.getenv('FIXED_STOP_EUR', '0') or 0.0),
        STOP_MAX_PCT=float(os.getenv('STOP_MAX_PCT', '0.3')),
        TARGET_ENTRY_COST=float(os.getenv('TARGET_ENTRY_COST', '10.0')),
        HARD_STOP_LOSS_PCT=float(os.getenv('HARD_STOP_LOSS_PCT', '0.15')),
        ENABLE_EOD_EXIT=parse_bool(os.getenv('ENABLE_EOD_EXIT', 'true'), True),
        EOD_EXIT_HOUR=int(os.getenv('EOD_EXIT_HOUR', '23')),
        STRUCTURE_LOOKBACK=int(os.getenv('STRUCTURE_LOOKBACK', '6')),
        STRUCTURE_BUFFER_ATR_K=float(os.getenv('STRUCTURE_BUFFER_ATR_K', '0.5')),
        MIN_STOP_ATR_K=float(os.getenv('MIN_STOP_ATR_K', '2.0')),
        MIN_VOL_MULT=float(os.getenv('MIN_VOL_MULT', '1.0')),
        MIN_BODY_PCT=float(os.getenv('MIN_BODY_PCT', '0.0')),
        COOLDOWN_BARS=int(os.getenv('COOLDOWN_BARS', '0')),
        MAX_TRADES_PER_DAY=int(os.getenv('MAX_TRADES_PER_DAY', '0')),
        MIN_RR=float(os.getenv('MIN_RR', '2.0')),
        CCXT_TIMEOUT_MS=int(os.getenv('CCXT_TIMEOUT_MS', '30000')),
        CCXT_RETRY_COUNT=int(os.getenv('CCXT_RETRY_COUNT', '3')),
        CCXT_RETRY_BACKOFF_SEC=float(os.getenv('CCXT_RETRY_BACKOFF_SEC', '1.0')),
        ENABLE_TRAIL_PROFIT=parse_bool(os.getenv('ENABLE_TRAIL_PROFIT', 'false'), False),
        TRAIL_PROFIT_TRIGGER_PCT=float(os.getenv('TRAIL_PROFIT_TRIGGER_PCT', '0.02')),
        TRAIL_PROFIT_OFFSET_PCT=float(os.getenv('TRAIL_PROFIT_OFFSET_PCT', '0.01')),
        TRAIL_PROFIT_MIN_LOCK_PCT=float(os.getenv('TRAIL_PROFIT_MIN_LOCK_PCT', '0.001')),
        TRAIL_PROFIT_MIN_MOVE_PCT=float(os.getenv('TRAIL_PROFIT_MIN_MOVE_PCT', '0.002')),
        TRAIL_PROFIT_STOP_LIMIT_GAP_PCT=float(os.getenv('TRAIL_PROFIT_STOP_LIMIT_GAP_PCT', '0.001')),
    )
    return cfg
