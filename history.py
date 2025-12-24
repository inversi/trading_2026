# -*- coding: utf-8 -*-
# Выгрузка истории сделок и ордеров.
from datetime import datetime, timezone, timedelta
import os
from typing import List, Optional

import pandas as pd

from trading_2026.exchange import Exchange
from trading_2026.logging_utils import log, log_error


def collect_history_last_month(ex: Exchange, symbols: Optional[List[str]] = None, days: int = 30, out_dir: str = 'logs/history') -> None:
    os.makedirs(out_dir, exist_ok=True)
    since_dt = datetime.now(timezone.utc) - timedelta(days=days)
    since_ms = int(since_dt.timestamp() * 1000)

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
