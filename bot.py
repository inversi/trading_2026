# -*- coding: utf-8 -*-
# Точка входа и главный цикл бота.
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from trading_2026.config import load_config
from trading_2026.exchange import Exchange, filter_markets_by_affordability
from trading_2026.logging_utils import setup_logging, log, log_error
from trading_2026.strategy import BreakoutWithATRAndRSI


def main():
    cfg = load_config()
    setup_logging('logs')
    log(f"Запуск бота — режим={cfg.MODE} рынки={cfg.MARKETS} таймфрейм={cfg.TIMEFRAME}", True)

    if cfg.MODE == 'live' and not cfg.API_KEY:
        log('В режиме LIVE нужны API_KEY/API_SECRET в .env — остановка.', True)
        return

    ex = Exchange(cfg)
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
        tradable_markets = filter_markets_by_affordability(ex, cfg.MARKETS, eur_balance, verbose=True)
        if not tradable_markets:
            log("Нет рынков, удовлетворяющих минимальному требованию по стоимости ордера — анализ будет пропущен.")
        else:
            log(f"Будут анализироваться только рынки: {tradable_markets}")
        cfg.MARKETS = tradable_markets
    except Exception as e:
        log_error("Не удалось получить баланс/лимиты", e)

    strat = BreakoutWithATRAndRSI(cfg, ex)
    try:
        strat.bootstrap_existing_positions()
    except Exception as e:
        log_error("Не удалось инициализировать существующие позиции", e)

    last_heartbeat_ts = time.time()
    while True:
        try:
            strat.on_tick()
        except Exception as e:
            log_error('Ошибка внутри on_tick', e)
        now_ts = time.time()
        if now_ts - last_heartbeat_ts >= 120:
            log("Heartbeat: бот работает", True)
            last_heartbeat_ts = now_ts
        time.sleep(max(1, cfg.SLEEP_SEC))


if __name__ == '__main__':
    main()
