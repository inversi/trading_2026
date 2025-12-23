# binance_autotrader.py
# -*- coding: utf-8 -*-
# =============================================================
# Описание
# Этот скрипт — простой торговый бот для Binance (spot) на ccxt.
# Он читает параметры из файла .env и по 1-минутным свечам ищет лонг-сетап
# (пробой максимума предыдущей свечи с фильтрами ATR и, при FAST_MODE, RSI на HTF).
# Обязательно начните с MODE=paper и только после проверки переключайте на live.
#
# Ключевые переменные .env (пример см. README/пояснение ниже):
#   MODE                 paper | live
#   MARKETS              Список торговых пар через запятую, например: PEPE/EUR,SHIB/EUR
#   TIMEFRAME            Основной ТФ для сигналов, например: 1m
#   LOOKBACK             Кол-во свечей для загрузки индикаторов и сигналов
#   RISK_PCT             Риск на сделку в долях депозита (0.007 = 0.7%).
#                        В текущей реализации не используется: размер позиции
#                        считается от фиксированной суммы входа (~10 EUR на сделку).
#                        Поле зарезервировано под альтернативный режим расчёта.
#   ATR_K                Множитель ATR для начального стоп-лосса (логический стоп)
#   ATR_PCT_MIN/MAX      Допустимая волатильность в долях цены (фильтр)
#   MAX_DAILY_DD_PCT     Лимит дневной просадки (в долях), при превышении — новые входы стоп
#   MAX_LOSSES_IN_ROW    Лимит серии убыточных сделок подряд (используется в on_tick)
#   FAST_MODE            Если true — включается MTF-фильтр тренда и объёма
#   FAST_HTF             Старший ТФ для фильтра (например 5m)
#   FAST_RSI_MIN         Минимальный RSI на HTF
#   FAST_MIN_VOL_SMA     Фильтр по объёму на 1m относительно SMA(объём)
#   EXCHANGE             Биржа (в этом примере только binance)
#   API_KEY / API_SECRET Ключи API для live-режима
#   TP_R_MULT            Тейк-профит в R (2.0 = 2R), если USE_TP=true
#   USE_TP               Ставить ли тейк-профит
#   USE_OCO_WHEN_AVAILABLE  Пытаться использовать OCO (тейк+стоп одним набором)
# =============================================================
# -------------------------------------------------------------
# Подробные комментарии:
# Ниже в коде добавлены пояснения к ключевым функциям, классам и
# участкам логики. Они должны помочь быстрее разобраться в том,
# как устроен бот, за что отвечает каждый блок и где безопасно
# вносить изменения под свои нужды.
# -------------------------------------------------------------
#
# -------------------------------------------------------------
# Краткое содержание основных компонентов
# -------------------------------------------------------------
# Функции индикаторов:
#   ema, sma       — экспоненциальная и простая скользящие средние
#   rsi            — индекс относительной силы (RSI) по Уайлдеру
#   atr            — средний истинный диапазон (ATR) по Уайлдеру
#
# Конфигурация и утилиты:
#   Config         — dataclass с параметрами бота (.env)
#   Position       — структура одной позиции (symbol, qty, entry, stop, tp)
#   load_config    — загрузка и разбор настроек из .env
#   parse_bool     — чтение булевых флагов из переменных окружения
#   fmt_float      — форматирование чисел для логов
#   format_ctx     — приведение контекста сигнала к строковому виду
#
# Логирование:
#   setup_logging  — настройка файловых логов (app/trades/errors)
#   log, log_info  — информационные сообщения
#   log_trade      — журнал входов/выходов и ордеров
#   log_error      — ошибки и исключения со стеком
#
# Обёртка биржи (Exchange):
#   fetch_ohlcv                — загрузка свечей OHLCV
#   balance_total_in_quote     — оценка всего спот-баланса в выбранной котируемой валюте
#   quote_free, base_free      — свободный баланс в котируемой/базовой валюте
#   avg_buy_price              — средняя цена покупок по символу
#   fetch_open_orders          — получение открытых ордеров
#   min_order_cost_quote       — минимальная стоимость ордера (minNotional)
#   price_step, min_price      — шаг и минимальная разрешённая цена
#   affordable                 — проверка, достаточно ли депозита для minNotional
#   round_qty                  — округление количества по лимитам биржи
#   create_market_buy/sell     — рыночные ордера
#   create_limit_sell          — лимитный ордер на продажу (TP, используется при инициализации позиции)
#   create_stop_loss_limit     — стоп-лимитный ордер на продажу (SL, используется при инициализации позиции)
#   create_oco_sell            — создание OCO (TP + SL, используется при инициализации позиции)
#   cancel_order/_all_orders   — отмена ордеров
#
# Стратегия BreakoutWithATRAndRSI:
#   _signal                    — генерация сигнала на вход по пробою и фильтрам ATR/RSI/объёма
#   _position_size             — расчёт размера позиции от заданного риска
#   _place_orders              — рыночный вход + установка логических уровней стоп/TP (без биржевых защитных ордеров)
#   _has_position_or_pending   — проверка существующих позиций и ожидающих ордеров
#   bootstrap_existing_positions — подключение уже открытых вручную позиций
#   _cancel_position           — логическое закрытие позиции и принудительная продажа
#   on_tick                    — один цикл стратегии (сигналы, управление позициями)
#
# Точка входа:
#   main                       — запуск бота, префлайт-проверки и основной цикл
#
import os
import time
from datetime import datetime, date, timezone, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import ccxt
from ccxt.base.errors import DDoSProtection, ExchangeNotAvailable, NetworkError, RequestTimeout, InvalidOrder
import pandas as pd
import numpy as np
from dotenv import load_dotenv

import logging
from logging.handlers import RotatingFileHandler
import pathlib

# =========================
# Вспомогательные функции: индикаторы
# =========================
# Простейшие технические индикаторы.
# Все функции принимают pandas.Series и возвращают Series такой же длины.
# Важно: индикаторы не модифицируют входные данные, а создают новые столбцы.
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

# RSI по методу Уайлдера. Используется сглаженное скользящее среднее (EWMA)
# для усреднения положительных и отрицательных изменений цены.
# Возвращает значения от 0 до 100, где перекупленность/перепроданность
# зависят от выбранных порогов.
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    # RSI по Уайлдеру (сглаживание через EWMA)
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

# ATR (Average True Range) по Уайлдеру. Считается на базе high/low/close и
# измеряет средний «истинный» диапазон свечей, то есть фактическую
# волатильность инструмента. Используется для расчёта стоп-лосса и фильтров.
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()  # ATR по Уайлдеру

# =========================
# Модели данных
# =========================
# Конфигурация бота. Все параметры читаются из .env (см. функцию load_config).
# Здесь собраны настройки режима (paper/live), риск-менеджмент, фильтры,
# параметры индикаторов и подключения к бирже.
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
    # Новые настраиваемые параметры
    TARGET_ENTRY_COST: float = 10.0
    HARD_STOP_LOSS_PCT: float = 0.15
    ENABLE_EOD_EXIT: bool = True
    EOD_EXIT_HOUR: int = 23
    # Фильтры против шума и переторговки
    STRUCTURE_LOOKBACK: int = 6          # сколько свечей назад искать локальный минимум для стопа
    STRUCTURE_BUFFER_ATR_K: float = 0.5  # буфер под локальным минимумом в долях ATR
    MIN_STOP_ATR_K: float = 2.0          # минимальная дистанция стопа в ATR
    MIN_VOL_MULT: float = 1.0           # минимальный множитель объёма к SMA объёма (1.0 = без фильтра)
    MIN_BODY_PCT: float = 0.0           # минимальный размер тела свечи в % цены (0.0 = без фильтра)
    COOLDOWN_BARS: int = 0              # пауза после выхода по символу (в барах 1m)
    MAX_TRADES_PER_DAY: int = 0         # лимит сделок в день (0 = без лимита)
    MIN_RR: float = 2.0                 # минимальный риск/прибыль, если USE_TP=true
    # Сеть/ccxt
    CCXT_TIMEOUT_MS: int = 30000
    CCXT_RETRY_COUNT: int = 3
    CCXT_RETRY_BACKOFF_SEC: float = 1.0
    # Трейлинг фиксации прибыли (аналог trailing stop, но в терминах % прибыли от входа).
    # Пример: при прибыли +2% ставим стоп на +1% (TRAIL_PROFIT_TRIGGER_PCT=0.02, TRAIL_PROFIT_OFFSET_PCT=0.01),
    # далее стоп двигается вверх, фиксируя потенциальную прибыль.
    ENABLE_TRAIL_PROFIT: bool = False
    TRAIL_PROFIT_TRIGGER_PCT: float = 0.02
    TRAIL_PROFIT_OFFSET_PCT: float = 0.01
    TRAIL_PROFIT_MIN_LOCK_PCT: float = 0.001
    TRAIL_PROFIT_MIN_MOVE_PCT: float = 0.002
    TRAIL_PROFIT_STOP_LIMIT_GAP_PCT: float = 0.001

# Структура для хранения информации об одной позиции.
# В данном примере бот торгует только в лонг, поэтому side='long'.
# qty  — фактическое количество базовой валюты
# entry — цена входа, stop — уровень стоп-лосса, tp — уровень тейк-профита.
@dataclass
class Position:
    symbol: str
    side: str           # сторона сделки (в примере только 'long')
    qty: float
    entry: float
    stop: float
    tp: Optional[float]
    trail_stop: Optional[float] = None

# =========================
# Вспомогательные утилиты
# =========================
# Утилита для чтения булевых флагов из переменных окружения.
# Поддерживаются значения вроде 'true', '1', 'yes' и т.п.
def parse_bool(v: str, default=False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in ('1', 'true', 'yes', 'y', 'on')

#
# Читает .env и формирует объект Config. Значения по умолчанию позволяют быстро
# запустить бота в paper-режиме. В live-режиме обязательны API_KEY / API_SECRET.
def load_config() -> Config:
    load_dotenv()
    MODE = os.getenv('MODE', 'paper').strip().lower()
    markets_env = os.getenv('MARKETS', 'BTC/EUR').replace(' ', '')
    MARKETS = [m for m in markets_env.split(',') if m]
    cfg = Config(
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

# Вспомогательная функция для получения текущего времени в UTC в виде строки
# для логов, когда структурный логгер ещё не инициализирован.
def now_ts() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

# Обёртка над log_info: позволяет условно подавлять вывод в зависимости
# от флага verbose (используется для «болтливого» режима).
def log(msg: str, verbose=True):
    if verbose:
        log_info(msg)

# Форматирование float без экспоненциальной нотации, чтобы значения
# в логах было удобно читать (особенно для мелких цен/объёмов).
def fmt_float(x: float, digits: int = 8) -> str:
    try:
        return ("{0:." + str(digits) + "f}").format(float(x))
    except Exception:
        return str(x)

# Преобразование словаря контекста сигнала так, чтобы все float были
# отформатированы в строки с фиксированной точностью. Удобно для логов.
def format_ctx(ctx: dict, digits: int = 8) -> dict:
    """Возвращает копию словаря ctx, где все значения float отформатированы как строки с фиксированным числом знаков."""
    out = {}
    for k, v in ctx.items():
        if isinstance(v, float):
            out[k] = fmt_float(v, digits)
        else:
            out[k] = v
    return out

# Преобразует контекст в компактную строку вида key=value, key2=value2.
# Используется при логировании условий входа: видно, почему сигнал сработал
# или почему не прошёл фильтры.
def format_ctx_str(ctx: dict, digits: int = 8) -> str:
    """Удобное однострочное представление вида key=value с числами в фиксированном формате."""
    f = format_ctx(ctx, digits)
    parts = []
    for k, v in f.items():
        parts.append(f"{k}={v}")
    return "{" + ", ".join(parts) + "}"

# =========================
# Вспомогательные функции выбора рынков
# =========================
# Фильтр списка рынков по «доступности».
# Оставляет только те инструменты, по которым текущий депозит позволяет
# выставить минимальный ордер (minNotional). Это уменьшает количество
# бессмысленных попыток входа с недостаточным балансом.
def filter_markets_by_affordability(ex: "Exchange", markets: List[str], total_quote_balance: float, verbose: bool = True) -> List[str]:
    """
    Возвращает только те символы, по которым на аккаунте достаточно средств в котируемой валюте,
    чтобы выполнить минимальный ордер (minNotional, минимальная стоимость ордера).
    Проверка эквивалентна вызову `ex.affordable`.

    Параметры
    ----------
    ex : Exchange
        Инициализированная обёртка над биржей.
    markets : List[str]
        Кандидаты символов для торговли.
    total_quote_balance : float
        Баланс в котируемой валюте, используемый для проверки доступности (например, EUR),
        обычно тот же, что выводится в префлайт-логе в main().
    verbose : bool
        Если True, логирует, какие рынки оставлены или исключены.
    """
    tradable = []
    for sym in markets:
        try:
            ok, need = ex.affordable(sym, total_quote_balance)
        except Exception as e:
            if verbose:
                log(f"{sym}: проверка доступности не удалась: {e}")
            continue
        if ok:
            tradable.append(sym)
            if verbose:
                log(f"{sym}: включено в анализ (достаточно средств для мин. ордера)")
        else:
            if verbose:
                if need is None:
                    log(f"{sym}: исключено из анализа — нельзя определить мин. стоимость ордера")
                else:
                    log(f"{sym}: исключено из анализа — недостаточно средств для мин. ордера (нужно ≈ {need:.2f})")
    return tradable

# =========================
# Структурированное логирование
# =========================
LOGGER_APP = None
LOGGER_TRADES = None
LOGGER_ERRORS = None


# Инициализация структурированного логирования.
# Создаются три отдельных лог-файла:
#  - app.log    — общий ход работы бота
#  - trades.log — записи о входах/выходах и ордерах
#  - errors.log — стеки исключений и критические ошибки
# В дополнение к файлам информация дублируется в консоль.
def setup_logging(base_dir: str = 'logs'):
    global LOGGER_APP, LOGGER_TRADES, LOGGER_ERRORS
    log_dir = pathlib.Path(base_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Ограничение размера одного файла лога (например, для отправки/загрузки сервисами с лимитом 10MB).
    # Дефолт: 9MB, чтобы гарантированно не превысить 10_485_760 байт.
    max_bytes = int(os.getenv("LOG_MAX_BYTES", 9 * 1024 * 1024))
    backup_count = int(os.getenv("LOG_BACKUP_COUNT", 30))

    # Общий форматтер с временем в UTC
    formatter = logging.Formatter('[%(asctime)s UTC] %(message)s')
    formatter.converter = time.gmtime  # принудительно используем UTC при форматировании времени

    def make_handler(filename: str, level: int) -> RotatingFileHandler:
        # Пишем в файл log_dir/filename, ротация по размеру: файлы не разрастаются больше max_bytes.
        # Резервные файлы будут называться app.log.1, app.log.2, ...
        h = RotatingFileHandler(
            log_dir / filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        h.setLevel(level)
        h.setFormatter(formatter)
        return h

    # Логгер общих событий приложения
    LOGGER_APP = logging.getLogger('app')
    LOGGER_APP.setLevel(logging.INFO)
    LOGGER_APP.handlers.clear()
    LOGGER_APP.addHandler(make_handler('app.log', logging.INFO))

    # Логгер сделок (входы/выходы и результаты постановки ордеров)
    LOGGER_TRADES = logging.getLogger('trades')
    LOGGER_TRADES.setLevel(logging.INFO)
    LOGGER_TRADES.handlers.clear()
    LOGGER_TRADES.addHandler(make_handler('trades.log', logging.INFO))

    # Логгер ошибок
    LOGGER_ERRORS = logging.getLogger('errors')
    LOGGER_ERRORS.setLevel(logging.ERROR)
    LOGGER_ERRORS.handlers.clear()
    LOGGER_ERRORS.addHandler(make_handler('errors.log', logging.ERROR))

    # Консольный вывод для оперативного просмотра
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

# =========================
# Обёртка биржи (ccxt)
# =========================
# Обёртка над ccxt для конкретной биржи (здесь — Binance spot).
# Содержит методы для:
#  - загрузки свечей
#  - оценки баланса в котируемой валюте
#  - вычисления minNotional, шага цены и т.д.
#  - создания/отмены рыночных, лимитных и OCO-ордеров
# Этот слой абстракции позволяет, при желании, позже адаптировать код
# под другую биржу, изменив реализацию только этого класса.
class Exchange:
    def __init__(self, cfg: Config):
        # Инициализация клиента ccxt для Binance spot. Загружаем рынки и
        # проверяем поддержку OCO (на некоторых аккаунтах/регионах может отличаться).
        if cfg.EXCHANGE != 'binance':
            raise ValueError('В этом примере поддерживается только Binance.')
        self.cfg = cfg
        self.ccxt = ccxt.binance({
            'apiKey': cfg.API_KEY or '',
            'secret': cfg.API_SECRET or '',
            'enableRateLimit': True,
            'timeout': int(getattr(cfg, 'CCXT_TIMEOUT_MS', 30000)),
            'options': {'defaultType': 'spot'}
        })
        # Загружаем унифицированные обозначения символов вида 'PEPE/EUR'
        self.markets = self._call_with_retries("load_markets", self.ccxt.load_markets)
        self.has_oco = bool(getattr(self.ccxt, 'has', {}).get('createOrderOCO', False))

    def _call_with_retries(self, name: str, fn, *args, **kwargs):
        retry_count = max(1, int(getattr(self.cfg, "CCXT_RETRY_COUNT", 3) or 1))
        base_backoff = float(getattr(self.cfg, "CCXT_RETRY_BACKOFF_SEC", 1.0) or 0.0)
        transient = (RequestTimeout, NetworkError, ExchangeNotAvailable, DDoSProtection)
        last_exc = None
        for attempt in range(1, retry_count + 1):
            try:
                return fn(*args, **kwargs)
            except transient as e:
                last_exc = e
                if attempt >= retry_count:
                    raise
                sleep_s = max(0.0, base_backoff * attempt)
                if sleep_s > 0:
                    time.sleep(sleep_s)
            except Exception as e:
                last_exc = e
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"{name}: неизвестная ошибка")

    # Загружает OHLCV-данные и превращает их в DataFrame с удобными именами
    # колонок. Именно с этим форматом далее работает стратегия.
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        # Загружаем OHLCV и приводим к DataFrame с колонками ts/open/high/low/close/volume.
        ohlcv = self._call_with_retries(
            f"fetch_ohlcv({symbol},{timeframe},{limit})",
            self.ccxt.fetch_ohlcv,
            symbol,
            timeframe=timeframe,
            limit=limit,
        )
        df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df

    def fetch_balance(self) -> dict:
        """fetch_balance с ретраями (может бросать исключение при окончательном провале)."""
        return self._call_with_retries("fetch_balance", self.ccxt.fetch_balance)

    # Оценка общего спот-баланса в выбранной котируемой валюте (по умолчанию EUR).
    # Перебираются все активы кошелька и для каждого ищется путь конверсии
    # в котируемую валюту (прямо, через USDT или через BTC). Нельзя использовать для
    # точной финансовой отчётности, но достаточно для risk-менеджмента.
    def balance_total_in_quote(self, quote: str = 'EUR') -> float:
        """Оценивает общий спот-баланс в выбранной котируемой валюте.

        Стратегия пересчёта:
          1) Прямая пара ASSET/QUOTE (или QUOTE/ASSET с обращением курса)
          2) Через USDT: ASSET/USDT и USDT/QUOTE (в любом направлении)
          3) Через BTC:  ASSET/BTC и BTC/QUOTE (в любом направлении)

        Если ни один путь не найден, актив пропускается при оценке.
        Используются балансы 'total' (free + used) спотового кошелька.
        """
        def _pair_last(base: str, q: str) -> Optional[float]:
            # Пробуем пару BASE/QUOTE, иначе QUOTE/BASE с обращением курса
            pair = f"{base}/{q}"
            try:
                t = self.ccxt.fetch_ticker(pair)
                px = float(t.get('last') or t.get('close') or ((t.get('bid') or 0)+(t.get('ask') or 0))/2 or 0.0)
                if px > 0:
                    return px
            except Exception:
                pass
            # Пробуем обратную пару
            pair_rev = f"{q}/{base}"
            try:
                t = self.ccxt.fetch_ticker(pair_rev)
                px = float(t.get('last') or t.get('close') or ((t.get('bid') or 0)+(t.get('ask') or 0))/2 or 0.0)
                if px > 0:
                    return 1.0 / px if px != 0 else None
            except Exception:
                pass
            return None

        def _asset_in_quote(asset: str, q: str) -> Optional[float]:
            if asset == q:
                return 1.0
            # 1) прямой курс
            px = _pair_last(asset, q)
            if px is not None:
                return px
            # 2) через USDT
            via = 'USDT'
            px1 = _pair_last(asset, via)
            px2 = _pair_last(via, q)
            if (px1 is not None) and (px2 is not None):
                return px1 * px2
            # 3) через BTC
            via = 'BTC'
            px1 = _pair_last(asset, via)
            px2 = _pair_last(via, q)
            if (px1 is not None) and (px2 is not None):
                return px1 * px2
            return None

        try:
            bal = self.ccxt.fetch_balance()
        except Exception:
            return 0.0

        totals = bal.get('total') or {}
        total_value = 0.0
        for asset, amt in totals.items():
            try:
                amount = float(amt or 0.0)
            except Exception:
                amount = 0.0
            if amount == 0.0:
                continue
            price_q = _asset_in_quote(asset, quote)
            if price_q is None:
                # не удалось оценить актив в выбранной котируемой валюте
                continue
            total_value += amount * price_q
        return float(total_value)

    # Оценка цены актива в BTC. Нужна для определения «пыли», которую можно
    # конвертировать в BNB через Binance Small Amount Exchange.
    def price_in_btc(self, asset: str) -> Optional[float]:
        asset = asset.upper()
        if asset == 'BTC':
            return 1.0
        pair_direct = f"{asset}/BTC"
        pair_reverse = f"BTC/{asset}"
        for pair, invert in ((pair_direct, False), (pair_reverse, True)):
            try:
                t = self.ccxt.fetch_ticker(pair)
                px = float(t.get('last') or t.get('close') or ((t.get('bid') or 0) + (t.get('ask') or 0)) / 2 or 0.0)
                if px > 0:
                    return (1.0 / px) if invert else px
            except Exception:
                continue
        return None

    # Возвращает стоимость указанного количества актива в BTC (или None, если цену не удалось получить).
    def value_in_btc(self, asset: str, amount: float) -> Optional[float]:
        px = self.price_in_btc(asset)
        if px is None:
            return None
        try:
            return float(amount) * float(px)
        except Exception:
            return None

    # Подбор активов, пригодных для Binance Small Amount Exchange (dust to BNB).
    # Фильтрует свободные остатки, которые оцениваются дешевле max_value_btc и не входят в список исключений.
    def find_small_balances_for_bnb(self, max_value_btc: float = 0.001, min_free: float = 1e-8, exclude: Optional[List[str]] = None) -> List[str]:
        bal = self.ccxt.fetch_balance()
        free_bal = bal.get('free') or {}
        exclude_set = set(a.upper() for a in (exclude or ['BNB', 'BTC', 'USDT', 'BUSD', 'FDUSD', 'USDC', 'EUR']))
        candidates: List[str] = []
        for asset, amt in free_bal.items():
            try:
                amount = float(amt or 0.0)
            except Exception:
                continue
            if amount <= min_free:
                continue
            asset_u = asset.upper()
            if asset_u in exclude_set:
                continue
            value_btc = self.value_in_btc(asset_u, amount)
            if value_btc is None:
                continue
            if value_btc <= max_value_btc:
                candidates.append(asset_u)
        return candidates

    # Выполняет конверсию «пыли» в BNB через /sapi/v1/asset/dust.
    # По умолчанию собирает кандидатов автоматически, но можно передать явный список assets.
    def convert_small_balances_to_bnb(self, assets: Optional[List[str]] = None, max_value_btc: float = 0.001) -> dict:
        if self.cfg.MODE != 'live':
            raise RuntimeError("Конвертация «пыли» доступна только в live-режиме.")
        if not hasattr(self.ccxt, 'sapiPostAssetDust'):
            raise RuntimeError("ccxt/binance не поддерживает sapiPostAssetDust на этом аккаунте.")

        assets = assets or self.find_small_balances_for_bnb(max_value_btc=max_value_btc)
        assets = [a for a in assets if a and a.upper() != 'BNB']
        if not assets:
            return {}
        try:
            return self.ccxt.sapiPostAssetDust({'asset': assets})
        except Exception as e:
            log_error("Не удалось выполнить конвертацию «пыли» в BNB", e)
            return {}

    # Свободный (не зарезервированный ордерами) баланс в выбранной
    # котируемой валюте на споте.
    def quote_free(self, quote: str = 'EUR') -> float:
        bal = self.ccxt.fetch_balance()
        try:
            return float(bal.get('free', {}).get(quote, 0.0) or 0.0)
        except Exception:
            return 0.0

    # Максимальное количество базовой валюты, которое можно купить
    # на весь доступный свободный баланс в котируемой валюте с небольшим запасом,
    # чтобы не упереться в ошибку недостатка средств.
    def max_buy_qty(self, symbol: str, safety: float = 0.995) -> float:
        quote = symbol.split('/')[1]
        free_quote = self.quote_free(quote)
        px = self.last_price(symbol)
        if px <= 0 or free_quote <= 0:
            return 0.0
        raw = (free_quote / px) * max(0.0, min(1.0, safety))
        # Если расчётный объём неположительный — сразу пропускаем
        if raw <= 0:
            return 0.0
        # На части рынков (например, LINK/EUR) объём должен быть не меньше шага точности.
        # amount_to_precision может выбросить InvalidOrder, если объём слишком мал;
        # в этом случае просто сообщаем, что покупать нечего.
        try:
            return float(self.ccxt.amount_to_precision(symbol, raw))
        except Exception:
            log(f"{symbol}: amount_to_precision для max_buy_qty отклонил расчётный объём={raw:.12g} — вернём 0", True)
            return 0.0

    # Свободный остаток базовой валюты по символу (например, PEPE для PEPE/EUR).
    # Используется при инициализации для обнаружения уже существующих ручных позиций.
    def base_free(self, symbol: str) -> float:
        """Свободный остаток базовой валюты по символу (например, PEPE для PEPE/EUR)."""
        base = symbol.split('/')[0]
        try:
            bal = self.fetch_balance()
        except Exception:
            return 0.0
        try:
            return float(bal.get('free', {}).get(base, 0.0) or 0.0)
        except Exception:
            return 0.0

    # Приблизительная средняя цена покупок по символу за заданный период.
    # Нужна для того, чтобы при инициализации задать разумный уровень входа,
    # если позиция была открыта вручную до запуска бота.
    def avg_buy_price(self, symbol: str, lookback_days: int = 30) -> Optional[float]:
        """Оцениваем среднюю цену покупок по символу за последние N дней (spot).
        Возвращает None, если нет данных о трейдах. Если у Binance нет полной истории в рамках лимита — это приблизительная оценка.
        """
        try:
            since_ms = int((datetime.now(timezone.utc).timestamp() - lookback_days * 86400) * 1000)
            trades = self.ccxt.fetch_my_trades(symbol, since=since_ms)
            buys = [t for t in trades if str(t.get('side')) == 'buy']
            if not buys:
                return None
            cost = 0.0
            amount = 0.0
            for t in buys:
                px = float(t.get('price') or 0.0)
                qty = float(t.get('amount') or 0.0)
                if px > 0 and qty > 0:
                    cost += px * qty
                    amount += qty
            if amount <= 0:
                return None
            return cost / amount
        except Exception:
            return None

    # Обёртка над fetch_open_orders ccxt. Возвращает список всех открытых
    # ордеров (по символу или по всем рынкам), чтобы стратегия могла
    # синхронизировать внутреннее состояние с тем, что реально висит на бирже.
    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[dict]:
        """Возвращает список открытых ордеров (spot). Если symbol=None — по всем рынкам."""
        try:
            return self._call_with_retries("fetch_open_orders", self.ccxt.fetch_open_orders, symbol)
        except Exception:
            return []

    def market_info(self, symbol: str) -> dict:
        return self.markets[symbol]

    def last_price(self, symbol: str) -> float:
        """Последняя цена по тикеру (last)."""
        t = self._call_with_retries("fetch_ticker", self.ccxt.fetch_ticker, symbol)
        return float(t.get('last') or t.get('close') or t.get('bid') or 0.0)

    # Возвращает оценку минимальной стоимости ордера (minNotional) в котируемой валюте.
    # Сначала пытается взять limits.cost.min из описания рынка, при отсутствии —
    # оценивает как min_amount * last_price. Если информации нет, вернёт None.
    def min_order_cost_quote(self, symbol: str, fallback_price: Optional[float] = None) -> Optional[float]:
        """Минимальная стоимость ордера в котируемой валюте (quote), если доступно.
        Пытается взять limits.cost.min, иначе оценивает как min_amount * last_price.
        Возвращает None, если оценить нельзя.
        """
        m = self.market_info(symbol)
        # 1) Пробуем прямой min notional из limits.cost.min
        cost_limits = m.get('limits', {}).get('cost') or {}
        min_cost = cost_limits.get('min')
        if min_cost is not None:
            try:
                return float(min_cost)
            except Exception:
                pass
        # 2) Если нет — пробуем amount.min * last_price
        amt_limits = m.get('limits', {}).get('amount') or {}
        min_amount = amt_limits.get('min')
        if min_amount is None:
            # иногда min находится в m['precision'] — тогда округлим 1e-precision
            prec = m.get('precision', {}).get('amount')
            if isinstance(prec, int) and prec >= 0:
                min_amount = 10 ** (-prec)
        if min_amount is None:
            return None
        price = fallback_price if fallback_price is not None else self.last_price(symbol)
        if price <= 0:
            return None
        try:
            return float(min_amount) * float(price)
        except Exception:
            return None

    # Шаг цены (tickSize) для инструмента. На Binance сначала читается
    # PRICE_FILTER.tickSize, при отсутствии — шаг оценивается по precision.
    def price_step(self, symbol: str) -> float:
        """Возвращает минимальный шаг цены (tickSize) для символа, предпочитая Binance PRICE_FILTER.tickSize."""
        m = self.market_info(symbol)
        # Сначала пробуем биржевые фильтры
        try:
            for f in m.get('info', {}).get('filters', []):
                if f.get('filterType') == 'PRICE_FILTER':
                    ts = float(f.get('tickSize') or 0)
                    if ts > 0:
                        return ts
        except Exception:
            pass
        # Затем пытаемся оценить шаг по precision
        prec = (m.get('precision') or {}).get('price')
        if isinstance(prec, int) and prec >= 0:
            try:
                return float(10 ** (-prec))
            except Exception:
                pass
        # Последняя попытка
        lim_min = (m.get('limits', {}).get('price') or {}).get('min')
        try:
            return float(lim_min or 0.0)
        except Exception:
            return 0.0

    # Минимально допустимая цена для инструмента согласно правилам биржи.
    # Нужна при расчёте стоп-лимит и OCO ордеров, чтобы не отправлять
    # на биржу цены ниже разрешённого уровня.
    def min_price(self, symbol: str) -> float:
        """Возвращает минимально допустимую цену для символа, если доступна (иначе 0)."""
        m = self.market_info(symbol)
        try:
            for f in m.get('info', {}).get('filters', []):
                if f.get('filterType') == 'PRICE_FILTER':
                    mp = float(f.get('minPrice') or 0)
                    return mp
        except Exception:
            pass
        lim_min = (m.get('limits', {}).get('price') or {}).get('min')
        try:
            return float(lim_min or 0.0)
        except Exception:
            return 0.0

    # Определение допустимых границ цены согласно фильтру PERCENT_PRICE_BY_SIDE.
    # Binance проверяет цену относительно средневзвешенной цены за период avgPriceMins,
    # поэтому используем weightedAvgPrice из тикера, а не только last.
    def _percent_price_bounds(self, symbol: str, side: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Возвращает (min_allowed, max_allowed, ref_price) для PERCENT_PRICE_BY_SIDE, если доступно."""
        side_l = str(side).lower()
        m = self.market_info(symbol)
        filters = (m.get('info') or {}).get('filters', []) or []

        # Ищем мультипликаторы для нужной стороны (SELL -> ask*, BUY -> bid*).
        mult_down = None
        mult_up = None
        for f in filters:
            try:
                if f.get('filterType') == 'PERCENT_PRICE_BY_SIDE':
                    if side_l == 'sell':
                        mult_down = float(f.get('askMultiplierDown')) if f.get('askMultiplierDown') is not None else None
                        mult_up = float(f.get('askMultiplierUp')) if f.get('askMultiplierUp') is not None else None
                    else:
                        mult_down = float(f.get('bidMultiplierDown')) if f.get('bidMultiplierDown') is not None else None
                        mult_up = float(f.get('bidMultiplierUp')) if f.get('bidMultiplierUp') is not None else None
                    break
            except Exception:
                continue

        # Тикер нужен, чтобы взять weightedAvgPrice (ближе к биржевому расчёту avgPriceMins).
        ref_price = None
        try:
            t = self.ccxt.fetch_ticker(symbol)
        except Exception:
            t = None

        if t:
            info = t.get('info') or {}
            candidates = [
                info.get('weightedAvgPrice'),
                info.get('weightedAveragePrice'),
                t.get('average'),
                t.get('vwap'),
                t.get('last'),
                t.get('close'),
                t.get('bid'),
                t.get('ask'),
            ]
            for c in candidates:
                try:
                    if c is not None:
                        ref_price = float(c)
                        if ref_price > 0:
                            break
                except Exception:
                    continue

        if (ref_price is None) or (ref_price <= 0):
            # Последний шанс — last_price (может вызвать доп. запрос, но лучше, чем None).
            try:
                ref_price = self.last_price(symbol)
            except Exception:
                ref_price = None

        if (ref_price is None) or (ref_price <= 0) or (mult_down is None):
            return None, None, ref_price

        min_allowed = ref_price * mult_down
        max_allowed = (ref_price * mult_up) if (mult_up is not None) else None
        return min_allowed, max_allowed, ref_price

    # Проверка, достаточно ли средств в котируемой валюте для выполнения
    # минимального ордера по символу. Используется как при префлайт-проверке,
    # так и при фильтрации рынков.
    def affordable(self, symbol: str, quote_balance: float) -> Tuple[bool, Optional[float]]:
        """Проверка: достаточно ли средств в котируемой валюте для минимального ордера."""
        min_cost = self.min_order_cost_quote(symbol)
        if min_cost is None:
            return False, None
        return (quote_balance >= min_cost), float(min_cost)

    # Округление количества базовой валюты до требований биржи.
    # Учитывает precision (кол-во знаков), минимальный размер ордера и
    # особый случай целочисленных количеств (precision == 0), чтобы избежать
    # ошибок InvalidOrder со стороны ccxt/Binance.
    # Если после всех проверок количество получается ниже допустимого, возвратит 0.0.
    def round_qty(self, symbol: str, qty: float) -> float:
        """Округляет количество до точности биржи и учитывает минимально допустимый объём.

        Избегает ошибок ccxt InvalidOrder, проверяя случай precision == 0 (целочисленные объёмы)
        и явно заданный минимальный размер. Если после округления объём получается ниже
        минимально разрешённого, возвращает 0.0, чтобы вызывающий код мог пропустить сделку.
        """
        info = self.market_info(symbol)
        limits_amount = (info.get('limits') or {}).get('amount') or {}
        min_amount = limits_amount.get('min')
        prec = (info.get('precision') or {}).get('amount')

        # Если рынок требует целочисленные объёмы (precision == 0) и qty < 1 — пропускаем
        try:
            if isinstance(prec, int) and prec == 0 and qty < 1:
                log(f"{symbol}: объём {qty:.8f} < 1 при precision=0 (нужны целые значения) — пропускаем", True)
                return 0.0
        except Exception:
            pass

        # Сначала пытаемся округлить к точности биржи через helper ccxt
        try:
            qty_rounded = float(self.ccxt.amount_to_precision(symbol, qty))
        except Exception:
            # Если helper ругается (например, объём меньше шага), аккуратно пропускаем
            log(f"{symbol}: amount_to_precision отклонил объём={qty:.12g} — пропускаем", True)
            return 0.0

        # Применяем явный минимальный объём, если он указан
        if min_amount is not None:
            try:
                min_amount = float(min_amount)
            except Exception:
                min_amount = None
        if (min_amount is not None) and (qty_rounded < min_amount):
            log(f"{symbol}: округлённый объём {qty_rounded:.8f} < минимального {min_amount:.8f} — пропускаем", True)
            return 0.0

        return float(qty_rounded)

    # Рынок-покупка по текущей цене. Вся логика проверки размера/баланса
    # выполняется в стратегии до вызова этого метода.
    def create_market_buy(self, symbol: str, amount: float) -> dict:
        return self.ccxt.create_order(symbol, 'market', 'buy', amount)

    # Рынок-продажа по текущей цене. Используется при принудительном закрытии
    # позиции, чтобы продать весь доступный объём базовой валюты.
    def create_market_sell(self, symbol: str, amount: float, price_hint: Optional[float] = None) -> dict:
        qty = self.round_qty(symbol, amount)
        if qty <= 0:
            raise InvalidOrder(f"{symbol} amount {amount} ниже минимума после округления")

        px = price_hint or self.last_price(symbol)
        min_cost = self.min_order_cost_quote(symbol, fallback_price=px)
        if min_cost is not None and px > 0 and (qty * px < float(min_cost)):
            raise InvalidOrder(f"{symbol} notional {qty*px:.8f} ниже minNotional {float(min_cost):.8f}")

        amount_p = float(self.ccxt.amount_to_precision(symbol, qty))
        return self.ccxt.create_order(symbol, 'market', 'sell', amount_p)

    def safe_market_sell_all(self, symbol: str, price_hint: Optional[float] = None) -> bool:
        """Продаёт весь доступный объём базовой валюты, если он проходит minNotional/precision."""
        qty_free = self.base_free(symbol)
        qty = self.round_qty(symbol, qty_free)
        if qty <= 0:
            log(f"{symbol}: остаток {qty_free:.8f} слишком мал для продажи, пропускаем", True)
            return False

        px = price_hint or self.last_price(symbol)
        min_cost = self.min_order_cost_quote(symbol, fallback_price=px)
        if min_cost is not None and px > 0 and (qty * px < float(min_cost)):
            log(f"{symbol}: остаток {qty:.8f} * цена {px:.8f} ниже minNotional {float(min_cost):.8f}, пропускаем", True)
            return False

        self.create_market_sell(symbol, qty, price_hint=px)
        return True

    # Лимитный ордер на продажу (обычно для TP). Все значения приводятся
    # к нужной точности через ccxt.*_to_precision. Для Binance дополнительно
    # учитываем фильтр PERCENT_PRICE_BY_SIDE, чтобы не выходить за допустимый
    # диапазон цен относительно текущего рынка.
    def create_limit_sell(self, symbol: str, amount: float, price: float, params: dict=None) -> dict:
        # Binance валидирует цены относительно weightedAvgPrice*multiplier,
        # поэтому используем границы из фильтра PERCENT_PRICE_BY_SIDE.
        min_allowed, max_allowed, ref_price = self._percent_price_bounds(symbol, 'sell')
        if (min_allowed is not None) and (ref_price is not None) and ref_price > 0:
            try:
                price = max(price, min_allowed)
                if max_allowed is not None:
                    price = min(price, max_allowed)
            except Exception:
                # Если что-то пошло не так при обрезке — оставляем исходную цену,
                # пусть ccxt/биржа сообщит об ошибке.
                pass

        amount_p = float(self.ccxt.amount_to_precision(symbol, amount))
        price_p = float(self.ccxt.price_to_precision(symbol, price))
        return self.ccxt.create_order(symbol, 'limit', 'sell', amount_p, price_p, params or {})

    # Стоп-лимитный ордер на продажу для Binance spot. Биржа требует
    # одновременно указать stopPrice (триггер) и price (лимитная цена).
    def create_stop_loss_limit(self, symbol: str, amount: float, stop_price: float, limit_price: float) -> dict:
        # Binance spot stop-loss-limit:
        #  - type 'stop_loss_limit'
        #  - price (limit price) + param stopPrice
        #  - обе цены должны проходить PRICE_FILTER и PERCENT_PRICE_BY_SIDE
        min_allowed, max_allowed, ref_price = self._percent_price_bounds(symbol, 'sell')
        if (min_allowed is not None) and (ref_price is not None) and ref_price > 0:
            try:
                # Для SELL-ордеров Binance требует:
                #   stop_price, limit_price >= ref_price * askMultiplierDown
                #   stop_price, limit_price <= ref_price * askMultiplierUp
                stop_price = max(stop_price, min_allowed)
                limit_price = max(limit_price, min_allowed)
                if max_allowed is not None:
                    stop_price = min(stop_price, max_allowed)
                    limit_price = min(limit_price, max_allowed)
            except Exception:
                # В случае любой ошибки просто не трогаем исходные уровни
                pass

        amount_p = float(self.ccxt.amount_to_precision(symbol, amount))
        stop_p = float(self.ccxt.price_to_precision(symbol, stop_price))
        limit_p = float(self.ccxt.price_to_precision(symbol, limit_price))
        params = {
            'stopPrice': stop_p,
            'timeInForce': 'GTC',
        }
        return self.ccxt.create_order(symbol, 'stop_loss_limit', 'sell', amount_p, limit_p, params)

    # Создание OCO-ордера (One-Cancels-the-Other) через SAPI Binance.
    # Состоит из связки лимитного TP и стоп-лимит ордера. При исполнении
    # одного из них второй автоматически отменяется биржей.
    def create_oco_sell(self, symbol: str, amount: float, take_profit_price: float, stop_price: float, stop_limit_price: float) -> dict:
        # Binance OCO требует:
        #  - корректные precision
        #  - выполнение PRICE_FILTER и PERCENT_PRICE_BY_SIDE для всех цен (TP/SL)
        min_allowed, max_allowed, ref_price = self._percent_price_bounds(symbol, 'sell')
        if (min_allowed is not None) and (ref_price is not None) and ref_price > 0:
            try:
                # Ограничиваем все цены снизу/сверху допустимым диапазоном.
                take_profit_price = max(take_profit_price, min_allowed)
                stop_price = max(stop_price, min_allowed)
                stop_limit_price = max(stop_limit_price, min_allowed)
                if max_allowed is not None:
                    take_profit_price = min(take_profit_price, max_allowed)
                    stop_price = min(stop_price, max_allowed)
                    stop_limit_price = min(stop_limit_price, max_allowed)
            except Exception:
                pass

        symbol_id = self.ccxt.market_id(symbol) if hasattr(self.ccxt, 'market_id') else symbol.replace('/', '')
        params = {
            'symbol': symbol_id,
            'side': 'SELL',
            'quantity': self.ccxt.amount_to_precision(symbol, amount),
            'price': self.ccxt.price_to_precision(symbol, take_profit_price),
            'stopPrice': self.ccxt.price_to_precision(symbol, stop_price),
            'stopLimitPrice': self.ccxt.price_to_precision(symbol, stop_limit_price),
            'stopLimitTimeInForce': 'GTC',
        }
        return self.ccxt.sapiPostOrderOco(params)

    # Отмена одиночного ордера по его id. Обёртка над ccxt.cancel_order,
    # которая глушит исключения и всегда возвращает словарь.
    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> dict:
        """Отменяет одиночный ордер по идентификатору (опционально с указанием символа)."""
        try:
            return self.ccxt.cancel_order(order_id, symbol)
        except Exception:
            return {}

    # Массовая отмена всех открытых ордеров (опционально только по одному символу).
    # Проходит по списку открытых ордеров и пытается отменить каждый.
    def cancel_all_orders(self, symbol: Optional[str] = None) -> None:
        """Отменяет все открытые ордера, опционально только по одному символу."""
        try:
            open_orders = self.fetch_open_orders(symbol)
        except Exception:
            open_orders = []
        for o in open_orders:
            try:
                oid = o.get('id') or o.get('orderId') or (o.get('info') or {}).get('orderId')
                if oid:
                    self.cancel_order(str(oid), symbol)
            except Exception:
                continue

# =========================
# Сбор истории ордеров и сделок
# =========================
# Вспомогательная функция, которую можно вызвать вручную (например, из REPL
# или отдельного скрипта), чтобы выгрузить историю ордеров и сделок за
# последний месяц в CSV-файлы для последующего анализа.
#
# ВАЖНО: функция НЕ вызывается автоматически внутри main() или стратегии,
# чтобы не замедлять работу бота. Её нужно вызывать явно:
#
#   from bot import Exchange, load_config, collect_history_last_month
#   cfg = load_config()
#   ex = Exchange(cfg)
#   collect_history_last_month(ex, symbols=cfg.MARKETS)
#
def collect_history_last_month(ex: Exchange, symbols: Optional[List[str]] = None, days: int = 30, out_dir: str = 'logs/history') -> None:
    """
    Выгружает историю сделок и ордеров за последние `days` дней (по умолчанию ~месяц)
    и сохраняет их в CSV-файлы в каталоге `out_dir`.

    Параметры
    ----------
    ex : Exchange
        Инициализированная обёртка биржи.
    symbols : Optional[List[str]]
        Список символов для выгрузки. Если None — биржа вернёт историю по всем доступным символам.
        (Поддерживается не всеми биржами; для Binance spot работает.)
    days : int
        Количество дней истории, по умолчанию 30.
    out_dir : str
        Папка для сохранения CSV-файлов.
    """
    os.makedirs(out_dir, exist_ok=True)
    since_dt = datetime.now(timezone.utc) - timedelta(days=days)
    since_ms = int(since_dt.timestamp() * 1000)

    # Сбор сделок (trade history)
    try:
        if symbols:
            all_trades = []
            for sym in symbols:
                try:
                    ts = ex.ccxt.fetch_my_trades(sym, since=since_ms)
                    for t in ts:
                        if 'symbol' not in t:
                            t['symbol'] = sym
                    all_trades.extend(ts)
                except Exception as e:
                    log(f"{sym}: не удалось получить историю сделок: {e}", True)
            trades = all_trades
        else:
            trades = ex.ccxt.fetch_my_trades(since=since_ms)
    except Exception as e:
        log_error("Не удалось получить историю сделок за период", e)
        trades = []

    if trades:
        try:
            df_trades = pd.DataFrame(trades)
            trades_path = os.path.join(out_dir, f"trades_last_{days}d.csv")
            df_trades.to_csv(trades_path, index=False)
            log(f"История сделок сохранена в {trades_path} (строк: {len(df_trades)})")
        except Exception as e:
            log_error("Ошибка при сохранении истории сделок в CSV", e)
    else:
        log("История сделок за указанный период пуста или недоступна.", True)

    # Сбор ордеров (order history)
    try:
        if symbols:
            all_orders = []
            for sym in symbols:
                try:
                    osym = ex.ccxt.fetch_orders(sym, since=since_ms)
                    for o in osym:
                        if 'symbol' not in o:
                            o['symbol'] = sym
                    all_orders.extend(osym)
                except Exception as e:
                    log(f"{sym}: не удалось получить историю ордеров: {e}", True)
            orders = all_orders
        else:
            orders = ex.ccxt.fetch_orders(since=since_ms)
    except Exception as e:
        log_error("Не удалось получить историю ордеров за период", e)
        orders = []

    if orders:
        try:
            df_orders = pd.DataFrame(orders)
            orders_path = os.path.join(out_dir, f"orders_last_{days}d.csv")
            df_orders.to_csv(orders_path, index=False)
            log(f"История ордеров сохранена в {orders_path} (строк: {len(df_orders)})")
        except Exception as e:
            log_error("Ошибка при сохранении истории ордеров в CSV", e)
    else:
        log("История ордеров за указанный период пуста или недоступна.", True)


# =========================
# Логика стратегии
# =========================
# Основная логика стратегии.
# Стратегия ищет лонг-входы по пробою максимума предыдущей свечи с фильтрами
# по ATR (волатильность), RSI на старшем таймфрейме (если включён FAST_MODE)
# и объёму. Управляет позициями, стопами, тейк-профитами и дневными лимитами
# по просадке и серии убыточных сделок.
class BreakoutWithATRAndRSI:
    def __init__(self, cfg: Config, ex: Exchange):
        # Сохраняем конфиг и обёртку биржи, инициализируем состояние стратегии:
        # открытые позиции, дневную точку отсчёта по equity и счётчик
        # подряд идущих убыточных сделок.
        self.cfg = cfg
        self.ex = ex
        self.positions: Dict[str, Position] = {}
        self.realized_pnl_eur: float = 0.0
        self.daily_start_equity_eur: Optional[float] = None
        self.losses_in_row: int = 0
        self.current_date: date = date.today()
        self.trades_today: int = 0
        self.last_exit_bar_index: Dict[str, int] = {}  # индекс бара, на котором был выход
        self.dust_ignore: set = set()  # символы с dust-остатками ниже minNotional

    def _calc_trail_profit_stop(self, pos: Position, last_price: float) -> Optional[float]:
        if not self.cfg.ENABLE_TRAIL_PROFIT:
            return None
        if pos is None or pos.qty <= 0 or pos.entry <= 0 or last_price <= 0:
            return None
        if pos.side != 'long':
            return None

        pnl_pct = (last_price - pos.entry) / pos.entry
        if pnl_pct < float(self.cfg.TRAIL_PROFIT_TRIGGER_PCT):
            return None

        offset = float(self.cfg.TRAIL_PROFIT_OFFSET_PCT)
        locked_profit_pct = pnl_pct - offset
        min_lock = float(self.cfg.TRAIL_PROFIT_MIN_LOCK_PCT)
        locked_profit_pct = max(locked_profit_pct, min_lock)
        stop_price = pos.entry * (1.0 + locked_profit_pct)

        # Стоп всегда должен быть ниже текущей цены, иначе ордер закроется сразу.
        stop_price = min(stop_price, last_price * (1.0 - 1e-4))
        return float(stop_price)

    def _extract_order_stop_price(self, order: dict) -> Optional[float]:
        if not order:
            return None
        for k in ("stopPrice", "stop_price", "triggerPrice", "trigger_price"):
            v = order.get(k)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
        info = order.get("info") or {}
        for k in ("stopPrice", "stop_price", "triggerPrice", "trigger_price"):
            v = info.get(k)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
        return None

    def _find_open_stop_loss_sells(self, open_orders: List[dict]) -> List[dict]:
        out = []
        for o in open_orders or []:
            try:
                side = (o.get("side") or "").lower()
                otype = (o.get("type") or "").lower()
                if side == "sell" and ("stop" in otype):
                    out.append(o)
            except Exception:
                continue
        return out

    def _upsert_live_trail_stop_order(self, pos: Position, balances: dict, open_orders: List[dict], stop_price: float) -> None:
        if self.cfg.MODE != "live":
            return
        qty_free = self._get_base_free_from_balances(pos.symbol, balances)
        qty = self.ex.round_qty(pos.symbol, qty_free)
        if qty <= 0:
            return

        existing_stops = self._find_open_stop_loss_sells(open_orders)
        existing = existing_stops[0] if existing_stops else None
        existing_stop = self._extract_order_stop_price(existing) if existing else None

        min_move_abs = max(0.0, float(pos.entry) * float(self.cfg.TRAIL_PROFIT_MIN_MOVE_PCT))
        if (existing_stop is not None) and (stop_price <= float(existing_stop) + min_move_abs):
            return

        if existing:
            oid = existing.get("id") or existing.get("orderId") or (existing.get("info") or {}).get("orderId")
            if oid:
                self.ex.cancel_order(str(oid), pos.symbol)

        gap_pct = max(0.0, float(self.cfg.TRAIL_PROFIT_STOP_LIMIT_GAP_PCT))
        limit_price = float(stop_price) * (1.0 - gap_pct)
        minp = self.ex.min_price(pos.symbol)
        limit_price = max(float(minp), float(limit_price))
        stop_price = max(float(minp), float(stop_price))
        if limit_price >= stop_price:
            limit_price = max(float(minp), stop_price * (1.0 - 1e-4))

        try:
            self.ex.create_stop_loss_limit(pos.symbol, qty, stop_price=stop_price, limit_price=limit_price)
        except Exception as e:
            log_error(f"{pos.symbol}: не удалось обновить трейлинг-стоп ордер", e)

    def _update_trailing_profit(self, pos: Position, last_price: float, balances: dict, open_orders: List[dict]) -> None:
        stop_price = self._calc_trail_profit_stop(pos, last_price)
        if stop_price is None:
            return

        min_move_abs = max(0.0, float(pos.entry) * float(self.cfg.TRAIL_PROFIT_MIN_MOVE_PCT))
        if (pos.trail_stop is not None) and (stop_price <= float(pos.trail_stop) + min_move_abs):
            return

        pos.trail_stop = float(stop_price)
        pnl_pct = (last_price - pos.entry) / pos.entry
        log(f"{pos.symbol}: трейлинг-профит обновлён стоп={fmt_float(pos.trail_stop, 8)} pnl={pnl_pct*100:.2f}%", self.cfg.VERBOSE)
        self._upsert_live_trail_stop_order(pos, balances, open_orders, stop_price=float(pos.trail_stop))

    def _get_base_free_from_balances(self, symbol: str, balances: dict) -> float:
        """Извлекает свободный баланс базовой валюты из уже загруженного словаря балансов."""
        base = symbol.split('/')[0]
        try:
            return float(balances.get('free', {}).get(base, 0.0) or 0.0)
        except Exception:
            return 0.0

    def _get_quote_free_from_balances(self, symbol: str, balances: dict) -> float:
        """Извлекает свободный баланс котируемой валюты из уже загруженного словаря балансов."""
        quote = symbol.split('/')[1]
        try:
            return float(balances.get('free', {}).get(quote, 0.0) or 0.0)
        except Exception:
            return 0.0

    def _get_total_base_from_balances(self, symbol: str, balances: dict) -> float:
        """Возвращает сумму свободного и зарезервированного баланса по базовой валюте."""
        base = symbol.split('/')[0]
        try:
            free = float(balances.get('free', {}).get(base, 0.0) or 0.0)
            used = float(balances.get('used', {}).get(base, 0.0) or 0.0)
            return max(0.0, free + used)
        except Exception:
            return 0.0

    # Проверяет, есть ли уже позиция или открытые ордера по символу.
    def _has_position_or_pending(self, symbol: str, balances: dict, open_orders: List[dict]) -> bool:
        """Возвращает истину, если уже есть позиция (в трекере или по балансу) или есть открытые ордера по символу."""
        # 1. Уже учтённая позиция
        if symbol in self.positions and self.positions[symbol].qty > 0:
            return True
        # 2. Остаток базовой монеты (из pre-fetched balances)
        try:
            qty = self._get_base_free_from_balances(symbol, balances)
            if qty > 0:
                last = self.ex.last_price(symbol)
                min_cost = self.ex.min_order_cost_quote(symbol, fallback_price=last)
                if (min_cost is None) or (qty * last >= float(min_cost)):
                    return True
        except Exception:
            pass
        # 3. Ожидающие ордера на символ (из pre-fetched list)
        if open_orders:
            if self.cfg.VERBOSE:
                log(f"{symbol}: пропуск — найдены открытые ордера: {len(open_orders)}")
            return True
        return False

    def _sync_position_after_tp(self, symbol: str, ref_price: float, balances: dict, open_orders: List[dict]) -> bool:
        """Проверяет, не закрылась ли позиция по TP/вручную на бирже."""
        pos = self.positions.get(symbol)
        if pos is None:
            return False

        base = symbol.split('/')[0]
        free = float((balances.get('free') or {}).get(base, 0.0) or 0.0)
        used = float((balances.get('used') or {}).get(base, 0.0) or 0.0)
        total = free + used

        if open_orders:
            return False

        est_cost = total * max(1e-12, float(ref_price))
        min_cost = self.ex.min_order_cost_quote(symbol, fallback_price=float(ref_price))
        if total <= 0 or ((min_cost is not None) and est_cost < float(min_cost)):
            exit_px = pos.tp if pos.tp is not None else ref_price
            quote = symbol.split('/')[1] if '/' in symbol else 'КОТИРОВКА'
            pnl_quote = (exit_px - pos.entry) * pos.qty
            log_trade(f"ВЫХОД {symbol} направление=лонг причина=тейк_или_ручное закрытие цена={fmt_float(exit_px,8)} результат={pnl_quote:.4f} {quote}")
            self.positions.pop(symbol, None)
            self.losses_in_row = 0
            return True

        return False

    def _apply_exit_rules_by_pct(self, pos: Position, last_price: float) -> bool:
        """Правила выхода для позиции."""
        if pos is None or pos.qty <= 0 or pos.entry <= 0:
            return False

        pnl_pct = (last_price - pos.entry) / pos.entry

        if pnl_pct <= -self.cfg.HARD_STOP_LOSS_PCT:
            reason = f"жесткий_стоп_{self.cfg.HARD_STOP_LOSS_PCT*100:.0f}% результат={pnl_pct:.4f}"
            self._cancel_position(pos, reason=reason, exit_price=last_price)
            self.losses_in_row += 1
            return True

        if pos.trail_stop is not None and last_price <= float(pos.trail_stop):
            self._cancel_position(
                pos,
                reason=f"трейлинг_профит_стоп стоп={float(pos.trail_stop):.8f} результат={pnl_pct:.4f}",
                exit_price=last_price,
            )
            self.losses_in_row = 0
            return True

        if self.cfg.USE_TP and pos.tp is not None and last_price >= pos.tp:
            self._cancel_position(pos, reason=f"тейк_достигнут тейк={pos.tp:.8f} результат={pnl_pct:.4f}", exit_price=last_price)
            self.losses_in_row = 0
            return True

        if self.cfg.ENABLE_EOD_EXIT and pnl_pct > 0 and datetime.now().time().hour >= self.cfg.EOD_EXIT_HOUR:
            near_tp = not (self.cfg.USE_TP and pos.tp is not None) or (last_price >= pos.tp * 0.98)
            if near_tp:
                self._cancel_position(pos, reason=f"выход_в_конце_дня результат={pnl_pct:.4f}", exit_price=last_price)
                self.losses_in_row = 0
                return True

        return False

    def _force_close_loss_positions(self, threshold_pct: float, balances: dict) -> None:
        """Принудительно закрывает позиции, просившие больше порога."""
        if threshold_pct <= 0.0:
            return

        if balances is None:
            try:
                balances = self.ex.fetch_balance()
            except Exception:
                balances = {}

        for symbol in self.cfg.MARKETS:
            qty = self._get_total_base_from_balances(symbol, balances)
            if qty <= 0:
                self.dust_ignore.discard(symbol)
                continue

            pos = self.positions.get(symbol)
            entry_price = None
            working_qty = qty
            if pos and pos.qty > 0 and pos.entry > 0:
                entry_price = pos.entry
                working_qty = pos.qty
            else:
                try:
                    entry_price = self.ex.avg_buy_price(symbol)
                except Exception:
                    entry_price = None

            if entry_price is None or entry_price <= 0:
                if self.cfg.VERBOSE:
                    log(f"{symbol}: не удалось оценить вход для принудительного стопа, пропускаем.", True)
                continue

            try:
                last_price = self.ex.last_price(symbol)
            except Exception as e:
                log_error(f"{symbol}: не удалось получить цену для принудительного стопа", e)
                continue

            min_cost = self.ex.min_order_cost_quote(symbol, fallback_price=last_price)
            if min_cost is not None and last_price > 0:
                notional = qty * last_price
                if notional < float(min_cost):
                    # Dust: один раз отмечаем, дальше молча игнорируем до роста остатка.
                    self.dust_ignore.add(symbol)
                    continue
                self.dust_ignore.discard(symbol)

            pnl_pct = (last_price - entry_price) / entry_price
            if pnl_pct > -threshold_pct:
                continue

            reason = f"принудительный_жесткий_стоп_{threshold_pct*100:.0f}% результат={pnl_pct:.4f}"
            if pos:
                self._cancel_position(pos, reason=reason, exit_price=last_price)
            else:
                temp_pos = Position(symbol, 'long', working_qty, entry_price, entry_price * (1.0 - self.cfg.STOP_MAX_PCT), None)
                self.positions[symbol] = temp_pos
                self._cancel_position(temp_pos, reason=reason, exit_price=last_price)

            self.losses_in_row += 1

    def bootstrap_existing_positions(self):
        """Сканирует свободные остатки базовых валют для всех символов и, если они есть, добавляет их как активные позиции."""
        log("Подхватываем уже существующие позиции...")
        for symbol in self.cfg.MARKETS:
            try:
                base_qty = self.ex.base_free(symbol)
                if base_qty <= 0:
                    self.dust_ignore.discard(symbol)
                    continue

                last = self.ex.last_price(symbol)
                min_cost = self.ex.min_order_cost_quote(symbol, fallback_price=last)
                if min_cost is not None and last > 0:
                    notional = base_qty * last
                    if notional < float(min_cost):
                        if symbol not in self.dust_ignore:
                            log(f"{symbol}: остаток {base_qty:.8f} ниже minNotional, игнорируем dust", True)
                        self.dust_ignore.add(symbol)
                        continue
                    self.dust_ignore.discard(symbol)

                tf_df = self.ex.fetch_ohlcv(symbol, self.cfg.TIMEFRAME, max(60, self.cfg.LOOKBACK))
                df = tf_df.copy()
                df['atr'] = atr(df, 14)
                atr_val = float(df.iloc[-1]['atr'])
                entry = self.ex.avg_buy_price(symbol) or last

                stop = entry - self.cfg.ATR_K * atr_val
                tp = entry + self.cfg.TP_R_MULT * (entry - stop) if self.cfg.USE_TP else None

                self.positions[symbol] = Position(symbol, 'long', base_qty, entry, stop, tp)
                log(f"{symbol}: инициализация — обнаружено {base_qty} {symbol.split('/')[0]}, создана позиция вход={fmt_float(entry, 8)} стоп={fmt_float(stop, 8)}")
            except Exception as e:
                log_error(f"{symbol}: не удалось инициализировать символ", e)

    def _reset_daily_limits_if_new_day(self):
        today = date.today()
        if today != self.current_date:
            self.current_date = today
            self.daily_start_equity_eur = None
            self.losses_in_row = 0
            self.trades_today = 0
            log("Новый день — сбрасываем дневные лимиты.")

    def _ensure_daily_equity_baseline(self, quote='EUR'):
        if self.daily_start_equity_eur is None:
            try:
                self.daily_start_equity_eur = self.ex.balance_total_in_quote(quote)
            except Exception:
                self.daily_start_equity_eur = 0.0

    def _check_daily_dd_limit(self, quote='EUR') -> bool:
        self._ensure_daily_equity_baseline(quote)
        try:
            current = self.ex.balance_total_in_quote(quote)
        except Exception:
            return True
        if self.daily_start_equity_eur == 0:
            return True
        dd = (current - self.daily_start_equity_eur) / self.daily_start_equity_eur
        return dd >= -self.cfg.MAX_DAILY_DD_PCT

    def _signal(self, symbol: str, tf_df: pd.DataFrame, htf_df: Optional[pd.DataFrame]) -> Tuple[bool, Dict]:
        df = tf_df.copy()
        df['sma20'] = sma(df['close'], 20)
        df['atr'] = atr(df, 14)
        df['atr_pct'] = df['atr'] / df['close']
        df['vol_sma'] = sma(df['volume'], max(2, self.cfg.FAST_MIN_VOL_SMA))

        prev, last = df.iloc[-2], df.iloc[-1]
        cond_breakout = (last['close'] > last['sma20']) and (last['close'] > prev['high'])
        atr_ok = self.cfg.ATR_PCT_MIN <= last['atr_pct'] <= self.cfg.ATR_PCT_MAX

        # Импульс/тело свечи: фильтр по размеру тела относительно цены
        body_pct = abs(float(last['close']) - float(last['open'])) / max(1e-12, float(last['close']))
        body_ok = body_pct >= (self.cfg.MIN_BODY_PCT / 100.0)
        
        rsi_ok, vol_ok = True, True
        if self.cfg.FAST_MODE and htf_df is not None:
            hdf = htf_df.copy()
            hdf['rsi'] = rsi(hdf['close'], 14)
            rsi_ok = hdf['rsi'].iloc[-1] >= self.cfg.FAST_RSI_MIN
            vol_ok = last['volume'] >= (df['vol_sma'].iloc[-1] if not np.isnan(df['vol_sma'].iloc[-1]) else 0)
        else:
            # Даже без FAST_MODE можно потребовать объём выше средней (если MIN_VOL_MULT > 1)
            vsma = float(df['vol_sma'].iloc[-1] if not np.isnan(df['vol_sma'].iloc[-1]) else 0.0)
            if vsma > 0 and self.cfg.MIN_VOL_MULT > 1.0:
                vol_ok = float(last['volume']) >= vsma * float(self.cfg.MIN_VOL_MULT)

        will_long = bool(cond_breakout and atr_ok and rsi_ok and vol_ok and body_ok)
        ctx = {
            'last_close': float(last['close']),
            'prev_high': float(prev['high']),
            'atr': float(last['atr']),
            'atr_pct': float(last['atr_pct']),
            'rsi_ok': rsi_ok,
            'vol_ok': vol_ok,
            'body_ok': body_ok,
            'body_pct': float(body_pct),
            'cond_breakout': cond_breakout,
            'atr_ok': atr_ok
        }
        return will_long, ctx

    def _position_size(self, symbol: str, entry: float, balances: dict) -> float:
        target_cost = self.cfg.TARGET_ENTRY_COST
        if entry <= 0:
            return 0.0
        
        qty_raw = target_cost / float(entry)
        
        free_quote = self._get_quote_free_from_balances(symbol, balances)
        px = self.ex.last_price(symbol)
        qty_cap_balance = (free_quote / px) * 0.995 if px > 0 and free_quote > 0 else 0.0
        
        qty = min(qty_raw, qty_cap_balance)
        qty = self.ex.round_qty(symbol, qty)

        if qty <= 0: return 0.0

        min_cost = self.ex.min_order_cost_quote(symbol, fallback_price=entry)
        if min_cost is not None and (qty * entry < float(min_cost)):
            return 0.0
        return qty

    def _calc_stop_from_structure(self, symbol: str, tf_df: pd.DataFrame, entry: float, atr_val: float) -> float:
        """Считает стоп-лосс как минимум из ATR-стопа и стопа по структуре."""
        atr_stop = entry - (self.cfg.FIXED_STOP_EUR if (self.cfg.FIXED_STOP_EUR and self.cfg.FIXED_STOP_EUR > 0) else self.cfg.ATR_K * atr_val)

        # Локальный минимум за N предыдущих свечей (структура)
        lookback = max(2, int(self.cfg.STRUCTURE_LOOKBACK))
        try:
            recent_slice = tf_df.iloc[-(lookback + 1):-1]
            recent_low = float(recent_slice['low'].min())
        except Exception:
            recent_low = entry

        struct_stop = recent_low - float(self.cfg.STRUCTURE_BUFFER_ATR_K) * atr_val

        # Выбираем более широкий стоп (ниже по цене)
        stop = min(float(atr_stop), float(struct_stop))

        # Гарантируем, что стоп дальше рыночного шума: минимум MIN_STOP_ATR_K * ATR
        min_stop = entry - float(self.cfg.MIN_STOP_ATR_K) * atr_val
        stop = min(stop, float(min_stop))

        # Учитываем floor по STOP_MAX_PCT и minPrice
        minp = self.ex.min_price(symbol)
        stop_floor = max(minp, entry * (1.0 - self.cfg.STOP_MAX_PCT))
        stop = max(stop, stop_floor)
        return float(stop)

    def _place_orders(self, symbol: str, qty: float, entry: float, atr_val: float, tf_df: pd.DataFrame) -> Optional[Position]:
        stop_virtual = self._calc_stop_from_structure(symbol, tf_df, entry, atr_val)
        minp = self.ex.min_price(symbol)
        stop_floor = max(minp, entry * (1.0 - self.cfg.STOP_MAX_PCT))
        stop_virtual = max(stop_virtual, stop_floor)
        tp = entry + self.cfg.TP_R_MULT * (entry - stop_virtual) if self.cfg.USE_TP else None

        # Проверяем минимальное RR, если тейк включён
        if self.cfg.USE_TP and tp is not None:
            risk = max(1e-12, entry - stop_virtual)
            reward = max(0.0, tp - entry)
            rr = reward / risk if risk > 0 else 0.0
            if rr < float(self.cfg.MIN_RR):
                log(f"{symbol}: пропуск — RR {rr:.2f} ниже минимального {self.cfg.MIN_RR:.2f}", self.cfg.VERBOSE)
                return None

        if self.cfg.MODE == 'paper':
            return Position(symbol, 'long', qty, entry, stop_virtual, tp)

        # live-режим
        buy = self.ex.create_market_buy(symbol, qty)
        filled_price = float(buy.get('average', buy.get('price', entry)) or entry)
        
        # Пересчитываем уровни исходя из фактической цены исполнения
        stop_final = self._calc_stop_from_structure(symbol, tf_df, filled_price, atr_val)
        stop_final = max(stop_final, filled_price * (1.0 - self.cfg.STOP_MAX_PCT))
        tp_final = filled_price + self.cfg.TP_R_MULT * (filled_price - stop_final) if self.cfg.USE_TP else None

        return Position(symbol, 'long', float(buy['filled']), filled_price, stop_final, tp_final)

    def _cancel_position(self, pos: Position, reason: str, exit_price: Optional[float] = None):
        quote_ccy = pos.symbol.split('/')[1]
        est_exit_px = exit_price or self.ex.last_price(pos.symbol)
        est_pnl_quote = (est_exit_px - pos.entry) * pos.qty
        log_trade(f"ВЫХОД {pos.symbol} направление=лонг причина={reason} цена={fmt_float(est_exit_px,8)} результат={est_pnl_quote:.4f} {quote_ccy}")

        if self.cfg.MODE == 'live':
            try:
                self.ex.cancel_all_orders(pos.symbol)

                sold = self.ex.safe_market_sell_all(pos.symbol, price_hint=est_exit_px)
                if sold:
                    log(f"{pos.symbol}: принудительно продан остаток", True)
            except Exception as e:
                log_error(f"{pos.symbol}: не удалось продать остаток при выходе", e)

        self.positions.pop(pos.symbol, None)
        # фиксируем выход для паузы/кулдауна
        try:
            self.last_exit_bar_index[pos.symbol] = int(self._bar_index)
        except Exception:
            self.last_exit_bar_index[pos.symbol] = 0

    def on_tick(self):
        self._reset_daily_limits_if_new_day()
        if not self._check_daily_dd_limit('EUR'):
            log("Достигнут дневной лимит просадки — новые входы приостановлены.")
            return

        try:
            all_balances = self.ex.fetch_balance()
            all_open_orders = self.ex.fetch_open_orders()
        except Exception as e:
            log_error("Не удалось загрузить балансы/ордера в начале цикла", e)
            return

        self._force_close_loss_positions(self.cfg.HARD_STOP_LOSS_PCT, all_balances)

        for symbol in self.cfg.MARKETS:
            try:
                tf_df = self.ex.fetch_ohlcv(symbol, self.cfg.TIMEFRAME, self.cfg.LOOKBACK)
                # индекс бара для кулдауна (просто счётчик на основе длины df)
                self._bar_index = len(tf_df)
                htf_df = self.ex.fetch_ohlcv(symbol, self.cfg.FAST_HTF, max(60, int(self.cfg.LOOKBACK/5))) if self.cfg.FAST_MODE else None

                will_long, ctx = self._signal(symbol, tf_df, htf_df)
                last_close = ctx['last_close']
                atr_val = ctx['atr']

                symbol_open_orders = [o for o in all_open_orders if o.get('symbol') == symbol]

                if symbol in self.positions:
                    min_cost = self.ex.min_order_cost_quote(symbol, fallback_price=last_close)
                    if min_cost is not None and last_close > 0:
                        notional = self.positions[symbol].qty * last_close
                        if notional < float(min_cost):
                            if symbol not in self.dust_ignore:
                                log(f"{symbol}: позиция ниже minNotional, игнорируем dust", True)
                            self.dust_ignore.add(symbol)
                            self.positions.pop(symbol, None)
                            continue
                    self.dust_ignore.discard(symbol)

                    if self._sync_position_after_tp(symbol, last_close, all_balances, symbol_open_orders):
                        continue

                    live_px = self.ex.last_price(symbol)
                    self._update_trailing_profit(self.positions[symbol], live_px, all_balances, symbol_open_orders)
                    if self._apply_exit_rules_by_pct(self.positions[symbol], live_px):
                        continue
                    continue

                # Дневной лимит сделок
                if self.cfg.MAX_TRADES_PER_DAY and self.cfg.MAX_TRADES_PER_DAY > 0:
                    if self.trades_today >= int(self.cfg.MAX_TRADES_PER_DAY):
                        log(f"{symbol}: пропуск — дневной лимит сделок {self.cfg.MAX_TRADES_PER_DAY} достигнут", self.cfg.VERBOSE)
                        continue

                # Кулдаун после выхода по символу
                if self.cfg.COOLDOWN_BARS and self.cfg.COOLDOWN_BARS > 0:
                    last_exit_idx = self.last_exit_bar_index.get(symbol)
                    if last_exit_idx is not None and (self._bar_index - int(last_exit_idx)) < int(self.cfg.COOLDOWN_BARS):
                        log(f"{symbol}: пропуск — кулдаун {self.cfg.COOLDOWN_BARS} баров после выхода", self.cfg.VERBOSE)
                        continue

                if self.losses_in_row >= self.cfg.MAX_LOSSES_IN_ROW:
                    log(f"{symbol}: пропуск (серия убыточных {self.losses_in_row} >= {self.cfg.MAX_LOSSES_IN_ROW})", self.cfg.VERBOSE)
                    continue

                if will_long:
                    if self._has_position_or_pending(symbol, all_balances, symbol_open_orders):
                        continue

                    qty = self._position_size(symbol, last_close, all_balances)
                    if qty <= 0:
                        log(f"{symbol}: объём слишком мал", self.cfg.VERBOSE)
                        continue

                    pos = self._place_orders(symbol, qty, last_close, atr_val, tf_df)
                    if pos:
                        self.positions[symbol] = pos
                        log_trade(f"ВХОД {symbol} направление=лонг объём={pos.qty:.8f} вход={fmt_float(pos.entry, 8)} стоп={fmt_float(pos.stop, 8)} тейк={fmt_float(pos.tp, 8) if pos.tp else 'нет'}")
                        self.trades_today += 1
                    else:
                        log(f"{symbol}: не удалось выставить ордер", True)
                elif self.cfg.VERBOSE_CTX:
                    log(f"{symbol}: входа нет — ctx=" + format_ctx_str(ctx, 8))

            except (RequestTimeout, NetworkError, ExchangeNotAvailable, DDoSProtection) as e:
                # Сетевые/временные ошибки — обычно временные, не нужно засорять лог длинным traceback.
                log_error(f"Сеть/таймаут при обработке {symbol} в on_tick | исключение={e}")
                continue
            except Exception as e:
                log_error(f'Не удалось обработать {symbol} в on_tick', e)
                continue

# =========================
# Главный цикл
# =========================
# Точка входа в программу.
# 1) Загружает конфигурацию из .env
# 2) Настраивает логирование
# 3) Создаёт обёртку биржи и стратегию
# 4) Выполняет префлайт-проверку баланса и фильтрацию рынков
# 5) Инициализирует уже имеющиеся позиции
# 6) Запускает бесконечный цикл, в котором периодически вызывается on_tick().
def main():
    # Читаем конфиг из .env и подготавливаем все параметры стратегии.
    # Точка входа: читаем конфиг, проверяем наличие ключей для live, создаём биржу и стратегию.
    cfg = load_config()
    # Инициализация структурированных логов: app.log, trades.log, errors.log
    setup_logging('logs')
    log(f"Запуск бота — режим={cfg.MODE} рынки={cfg.MARKETS} таймфрейм={cfg.TIMEFRAME}", True)

    if cfg.MODE == 'live' and not cfg.API_KEY:
        log('В режиме LIVE нужны API_KEY/API_SECRET в .env — остановка.', True)
        return

    # Создаём обёртку над ccxt и загружаем информацию о рынках.
    ex = Exchange(cfg)
    # Предзапуск: баланс в котируемой валюте (EUR) и проверка достаточности средств
    # Префлайт: оцениваем общий баланс в EUR и проверяем, по каким рынкам
    # депозит позволяет выставлять минимальные ордера. Остальные рынки
    # сразу исключаются из анализа.
    try:
        eur_balance = ex.balance_total_in_quote('EUR')
        log(f"Оценка эквивалента баланса ≈ {eur_balance:.2f} EUR (по рын. котировкам)", True)
        for sym in cfg.MARKETS:
            ok, need = ex.affordable(sym, eur_balance)
            if need is None:
                log(f"{sym}: нельзя оценить минимальную стоимость ордера (нет данных о лимитах)")
            else:
                status = 'ДОСТАТОЧНО' if ok else 'НЕДОСТАТОЧНО'
                log(f"{sym}: мин. ордер ≈ {need:.2f} EUR — {status}")
        # Отфильтруем список рынков по достаточности средств и будем анализировать только их
        tradable_markets = filter_markets_by_affordability(ex, cfg.MARKETS, eur_balance, verbose=True)
        if not tradable_markets:
            log("Нет рынков, удовлетворяющих минимальному требованию по стоимости ордера — анализ будет пропущен.")
        else:
            log(f"Будут анализироваться только рынки: {tradable_markets}")
        cfg.MARKETS = tradable_markets
    except Exception as e:
        log_error("Не удалось получить баланс/лимиты", e)

    # Создаём экземпляр стратегии, которая будет обрабатывать тики.
    strat = BreakoutWithATRAndRSI(cfg, ex)

    # Подхватываем уже существующие спот-позиции по указанным рынкам
    # и, при возможности, автоматически выставляем для них защитные ордера.
    # Добавляем в управление уже имеющиеся активы на споте
    try:
        strat.bootstrap_existing_positions()
    except Exception as e:
        log_error("Не удалось инициализировать существующие позиции", e)

    # Главный бесконечный цикл: на каждом шаге вызывается on_tick(),
    # после чего делается пауза SLEEP_SEC секунд.
    # Главный цикл: вызываем on_tick() с паузой SLEEP_SEC.
    while True:
        try:
            strat.on_tick()
        except Exception as e:
            log_error('Ошибка внутри on_tick', e)
        time.sleep(max(1, cfg.SLEEP_SEC))

if __name__ == '__main__':
    main()
