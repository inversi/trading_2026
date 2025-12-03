# config.py
import os
from dataclasses import dataclass
from typing import List, Optional
from dotenv import load_dotenv

def parse_bool(v: str, default=False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in ('1', 'true', 'yes', 'y', 'on')

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

def load_config() -> Config:
    load_dotenv()
    MODE = os.getenv('MODE', 'paper').strip().lower()
    markets_env = os.getenv('MARKETS', 'BTC/EUR').replace(' ', '')
    MARKETS = [m for m in markets_env.split(',') if m]
    return Config(
        MODE=MODE,
        MARKETS=MARKETS,
        TIMEFRAME=os.getenv('TIMEFRAME', '1m'),
        LOOKBACK=int(os.getenv('LOOKBACK', '600')),
        RISK_PCT=float(os.getenv('RISK_PCT', '0.005')),
        ATR_K=float(os.getenv('ATR_K', '3.0')),
        ATR_PCT_MIN=float(os.getenv('ATR_PCT_MIN', '0.002')),
        ATR_PCT_MAX=float(os.getenv('ATR_PCT_MAX', '0.08')),
        MAX_DAILY_DD_PCT=float(os.getenv('MAX_DAILY_DD_PCT', '0.02')),
        MAX_LOSSES_IN_ROW=int(os.getenv('MAX_LOSSES_IN_ROW', '3')),
        VERBOSE=parse_bool(os.getenv('VERBOSE', 'true'), True),
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
    )