# -*- coding: utf-8 -*-
# Торговая стратегия и управление позициями.
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ccxt.base.errors import DDoSProtection, ExchangeNotAvailable, NetworkError, RequestTimeout

from trading_2026.config import Config, StrategyConfig
from trading_2026.exchange import Exchange
from trading_2026.indicators import sma, rsi, atr
from trading_2026.logging_utils import log, log_error, log_trade, fmt_float, format_ctx_str


@dataclass
class Position:
    symbol: str
    side: str
    qty: float
    entry: float
    stop: float
    tp: Optional[float]
    trail_stop: Optional[float] = None


class BreakoutWithATRAndRSI:
    def __init__(self, cfg: Config, scfg: StrategyConfig, ex: Exchange):
        self.cfg = cfg
        self.scfg = scfg
        self.ex = ex
        self.positions: Dict[str, Position] = {}
        self.realized_pnl_eur: float = 0.0
        self.daily_start_equity_eur: Optional[float] = None
        self.losses_in_row: int = 0
        self.current_date: date = date.today()
        self.trades_today: int = 0
        self.last_exit_bar_index: Dict[str, int] = {}
        self.last_exit_bar_ts: Dict[str, datetime] = {}
        self.dust_ignore: set = set()
        self._bar_ts: Optional[datetime] = None

    def _timeframe_seconds(self) -> Optional[int]:
        tf = str(self.scfg.TIMEFRAME or "").strip().lower()
        if not tf:
            return None
        digits = ""
        for ch in tf:
            if ch.isdigit():
                digits += ch
            else:
                break
        unit = tf[len(digits):] if digits else tf
        try:
            mult = int(digits) if digits else 1
        except Exception:
            return None
        units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
        base = units.get(unit)
        if base is None:
            return None
        return mult * base

    def _calc_trail_profit_stop(self, pos: Position, last_price: float) -> Optional[float]:
        if not self.scfg.ENABLE_TRAIL_PROFIT:
            return None
        if pos is None or pos.qty <= 0 or pos.entry <= 0 or last_price <= 0:
            return None
        if pos.side != 'long':
            return None

        pnl_pct = (last_price - pos.entry) / pos.entry
        if pnl_pct < float(self.scfg.TRAIL_PROFIT_TRIGGER_PCT):
            return None

        offset = float(self.scfg.TRAIL_PROFIT_OFFSET_PCT)
        locked_profit_pct = pnl_pct - offset
        min_lock = float(self.scfg.TRAIL_PROFIT_MIN_LOCK_PCT)
        locked_profit_pct = max(locked_profit_pct, min_lock)
        stop_price = pos.entry * (1.0 + locked_profit_pct)
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

        min_move_abs = max(0.0, float(pos.entry) * float(self.scfg.TRAIL_PROFIT_MIN_MOVE_PCT))
        if (existing_stop is not None) and (stop_price <= float(existing_stop) + min_move_abs):
            return

        if existing:
            oid = existing.get("id") or existing.get("orderId") or (existing.get("info") or {}).get("orderId")
            if oid:
                self.ex.cancel_order(str(oid), pos.symbol)

        gap_pct = max(0.0, float(self.scfg.TRAIL_PROFIT_STOP_LIMIT_GAP_PCT))
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

        min_move_abs = max(0.0, float(pos.entry) * float(self.scfg.TRAIL_PROFIT_MIN_MOVE_PCT))
        if (pos.trail_stop is not None) and (stop_price <= float(pos.trail_stop) + min_move_abs):
            return

        pos.trail_stop = float(stop_price)
        pnl_pct = (last_price - pos.entry) / pos.entry
        log(f"{pos.symbol}: трейлинг-профит обновлён стоп={fmt_float(pos.trail_stop, 8)} pnl={pnl_pct*100:.2f}%", self.cfg.VERBOSE)
        self._upsert_live_trail_stop_order(pos, balances, open_orders, stop_price=float(pos.trail_stop))

    def _get_base_free_from_balances(self, symbol: str, balances: dict) -> float:
        base = symbol.split('/')[0]
        try:
            return float(balances.get('free', {}).get(base, 0.0) or 0.0)
        except Exception:
            return 0.0

    def _get_quote_free_from_balances(self, symbol: str, balances: dict) -> float:
        quote = symbol.split('/')[1]
        try:
            return float(balances.get('free', {}).get(quote, 0.0) or 0.0)
        except Exception:
            return 0.0

    def _get_total_base_from_balances(self, symbol: str, balances: dict) -> float:
        base = symbol.split('/')[0]
        try:
            free = float(balances.get('free', {}).get(base, 0.0) or 0.0)
            used = float(balances.get('used', {}).get(base, 0.0) or 0.0)
            return max(0.0, free + used)
        except Exception:
            return 0.0

    def _has_position_or_pending(self, symbol: str, balances: dict, open_orders: List[dict]) -> bool:
        if symbol in self.positions and self.positions[symbol].qty > 0:
            return True
        try:
            qty = self._get_base_free_from_balances(symbol, balances)
            if qty > 0:
                last = self.ex.last_price(symbol)
                min_cost = self.ex.min_order_cost_quote(symbol, fallback_price=last)
                if (min_cost is None) or (qty * last >= float(min_cost)):
                    return True
        except Exception:
            pass
        if open_orders:
            if self.cfg.VERBOSE:
                log(f"{symbol}: пропуск — найдены открытые ордера: {len(open_orders)}")
            return True
        return False

    def _sync_position_after_tp(self, symbol: str, ref_price: float, balances: dict, open_orders: List[dict]) -> bool:
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
        if pos is None or pos.qty <= 0 or pos.entry <= 0:
            return False

        pnl_pct = (last_price - pos.entry) / pos.entry

        if pnl_pct <= -self.scfg.HARD_STOP_LOSS_PCT:
            reason = f"жесткий_стоп_{self.scfg.HARD_STOP_LOSS_PCT*100:.0f}% результат={pnl_pct:.4f}"
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

        if self.scfg.USE_TP and pos.tp is not None and last_price >= pos.tp:
            self._cancel_position(pos, reason=f"тейк_достигнут тейк={pos.tp:.8f} результат={pnl_pct:.4f}", exit_price=last_price)
            self.losses_in_row = 0
            return True

        if self.scfg.ENABLE_EOD_EXIT and pnl_pct > 0 and datetime.now().time().hour >= self.scfg.EOD_EXIT_HOUR:
            near_tp = not (self.scfg.USE_TP and pos.tp is not None) or (last_price >= pos.tp * 0.98)
            if near_tp:
                self._cancel_position(pos, reason=f"выход_в_конце_дня результат={pnl_pct:.4f}", exit_price=last_price)
                self.losses_in_row = 0
                return True

        return False

    def _force_close_loss_positions(self, threshold_pct: float, balances: dict) -> None:
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
                temp_pos = Position(symbol, 'long', working_qty, entry_price, entry_price * (1.0 - self.scfg.STOP_MAX_PCT), None)
                self.positions[symbol] = temp_pos
                self._cancel_position(temp_pos, reason=reason, exit_price=last_price)

            self.losses_in_row += 1

    def bootstrap_existing_positions(self):
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

                tf_df = self.ex.fetch_ohlcv(symbol, self.scfg.TIMEFRAME, max(60, self.scfg.LOOKBACK))
                df = tf_df.copy()
                df['atr'] = atr(df, 14)
                atr_val = float(df.iloc[-1]['atr'])
                entry = self.ex.avg_buy_price(symbol) or last

                stop = entry - self.scfg.ATR_K * atr_val
                tp = entry + self.scfg.TP_R_MULT * (entry - stop) if self.scfg.USE_TP else None

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
        return dd >= -self.scfg.MAX_DAILY_DD_PCT

    def _signal(self, symbol: str, tf_df: pd.DataFrame, htf_df: Optional[pd.DataFrame]) -> Tuple[bool, Dict]:
        df = tf_df.copy()
        df['sma20'] = sma(df['close'], 20)
        df['atr'] = atr(df, 14)
        df['atr_pct'] = df['atr'] / df['close']
        df['vol_sma'] = sma(df['volume'], max(2, self.scfg.FAST_MIN_VOL_SMA))

        prev, last = df.iloc[-2], df.iloc[-1]
        cond_breakout = (last['close'] > last['sma20']) and (last['close'] > prev['high'])
        atr_ok = self.scfg.ATR_PCT_MIN <= last['atr_pct'] <= self.scfg.ATR_PCT_MAX

        body_pct = abs(float(last['close']) - float(last['open'])) / max(1e-12, float(last['close']))
        body_ok = body_pct >= (self.scfg.MIN_BODY_PCT / 100.0)
        
        rsi_ok, vol_ok = True, True
        if self.scfg.FAST_MODE and htf_df is not None:
            hdf = htf_df.copy()
            hdf['rsi'] = rsi(hdf['close'], 14)
            rsi_ok = hdf['rsi'].iloc[-1] >= self.scfg.FAST_RSI_MIN
            vol_ok = last['volume'] >= (df['vol_sma'].iloc[-1] if not np.isnan(df['vol_sma'].iloc[-1]) else 0)
        else:
            vsma = float(df['vol_sma'].iloc[-1] if not np.isnan(df['vol_sma'].iloc[-1]) else 0.0)
            if vsma > 0 and self.scfg.MIN_VOL_MULT > 1.0:
                vol_ok = float(last['volume']) >= vsma * float(self.scfg.MIN_VOL_MULT)

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
        target_cost = self.scfg.TARGET_ENTRY_COST
        if entry <= 0:
            return 0.0

        qty_raw = target_cost / float(entry)

        free_quote = self._get_quote_free_from_balances(symbol, balances)
        px = self.ex.last_price(symbol)
        qty_cap_balance = (free_quote / px) * 0.995 if px > 0 and free_quote > 0 else 0.0

        qty = min(qty_raw, qty_cap_balance)
        qty = self.ex.round_qty(symbol, qty)

        if qty <= 0:
            return 0.0

        min_cost = self.ex.min_order_cost_quote(symbol, fallback_price=entry)
        if min_cost is not None and (qty * entry < float(min_cost)):
            return 0.0
        return qty

    def _calc_stop_from_structure(self, symbol: str, tf_df: pd.DataFrame, entry: float, atr_val: float) -> float:
        atr_stop = entry - (self.scfg.FIXED_STOP_EUR if (self.scfg.FIXED_STOP_EUR and self.scfg.FIXED_STOP_EUR > 0) else self.scfg.ATR_K * atr_val)

        lookback = max(2, int(self.scfg.STRUCTURE_LOOKBACK))
        try:
            recent_slice = tf_df.iloc[-(lookback + 1):-1]
            recent_low = float(recent_slice['low'].min())
        except Exception:
            recent_low = entry

        struct_stop = recent_low - float(self.scfg.STRUCTURE_BUFFER_ATR_K) * atr_val

        stop = min(float(atr_stop), float(struct_stop))

        min_stop = entry - float(self.scfg.MIN_STOP_ATR_K) * atr_val
        stop = min(stop, float(min_stop))

        minp = self.ex.min_price(symbol)
        stop_floor = max(minp, entry * (1.0 - self.scfg.STOP_MAX_PCT))
        stop = max(stop, stop_floor)
        return float(stop)

    def _place_orders(self, symbol: str, qty: float, entry: float, atr_val: float, tf_df: pd.DataFrame) -> Optional[Position]:
        stop_virtual = self._calc_stop_from_structure(symbol, tf_df, entry, atr_val)
        minp = self.ex.min_price(symbol)
        stop_floor = max(minp, entry * (1.0 - self.scfg.STOP_MAX_PCT))
        stop_virtual = max(stop_virtual, stop_floor)
        tp = entry + self.scfg.TP_R_MULT * (entry - stop_virtual) if self.scfg.USE_TP else None

        if self.scfg.USE_TP and tp is not None:
            risk = max(1e-12, entry - stop_virtual)
            reward = max(0.0, tp - entry)
            rr = reward / risk if risk > 0 else 0.0
            if rr < float(self.scfg.MIN_RR):
                log(f"{symbol}: пропуск — RR {rr:.2f} ниже минимального {self.scfg.MIN_RR:.2f}", self.cfg.VERBOSE)
                return None

        if self.cfg.MODE == 'paper':
            return Position(symbol, 'long', qty, entry, stop_virtual, tp)

        buy = self.ex.create_market_buy(symbol, qty)
        filled_price = float(buy.get('average', buy.get('price', entry)) or entry)

        stop_final = self._calc_stop_from_structure(symbol, tf_df, filled_price, atr_val)
        stop_final = max(stop_final, filled_price * (1.0 - self.scfg.STOP_MAX_PCT))
        tp_final = filled_price + self.scfg.TP_R_MULT * (filled_price - stop_final) if self.scfg.USE_TP else None

        return Position(symbol, 'long', float(buy['filled']), filled_price, stop_final, tp_final)

    def _cancel_position(self, pos: Position, reason: str, exit_price: Optional[float] = None):
        quote_ccy = pos.symbol.split('/')[1]
        est_exit_px = exit_price or self.ex.last_price(pos.symbol)
        est_pnl_quote = (est_exit_px - pos.entry) * pos.qty
        plan_profit = None
        if pos.tp is not None:
            plan_profit = max(0.0, (pos.tp - pos.entry) * pos.qty)
        plan_loss = max(0.0, (pos.entry - pos.stop) * pos.qty)
        plan_profit_str = f"{plan_profit:.4f} {quote_ccy}" if plan_profit is not None else "нет"
        plan_loss_str = f"{plan_loss:.4f} {quote_ccy}"
        log_trade(
            f"ВЫХОД {pos.symbol} направление=лонг причина={reason} цена={fmt_float(est_exit_px,8)} "
            f"результат={est_pnl_quote:.4f} {quote_ccy} план_прибыль={plan_profit_str} план_убыток={plan_loss_str}"
        )

        if self.cfg.MODE == 'live':
            try:
                self.ex.cancel_all_orders(pos.symbol)

                sold = self.ex.safe_market_sell_all(pos.symbol, price_hint=est_exit_px)
                if sold:
                    log(f"{pos.symbol}: принудительно продан остаток", True)
            except Exception as e:
                log_error(f"{pos.symbol}: не удалось продать остаток при выходе", e)

        self.positions.pop(pos.symbol, None)
        try:
            self.last_exit_bar_index[pos.symbol] = int(self._bar_index)
        except Exception:
            self.last_exit_bar_index[pos.symbol] = 0
        try:
            self.last_exit_bar_ts[pos.symbol] = self._bar_ts or datetime.utcnow()
        except Exception:
            self.last_exit_bar_ts[pos.symbol] = datetime.utcnow()

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

        self._force_close_loss_positions(self.scfg.HARD_STOP_LOSS_PCT, all_balances)

        for symbol in self.cfg.MARKETS:
            try:
                tf_df = self.ex.fetch_ohlcv(symbol, self.scfg.TIMEFRAME, self.scfg.LOOKBACK)
                self._bar_index = len(tf_df)
                if len(tf_df) > 0:
                    try:
                        self._bar_ts = tf_df.iloc[-1]["ts"].to_pydatetime()
                    except Exception:
                        self._bar_ts = None
                else:
                    self._bar_ts = None
                htf_df = self.ex.fetch_ohlcv(symbol, self.scfg.FAST_HTF, max(60, int(self.scfg.LOOKBACK / 5))) if self.scfg.FAST_MODE else None

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

                if self.scfg.MAX_TRADES_PER_DAY and self.scfg.MAX_TRADES_PER_DAY > 0:
                    if self.trades_today >= int(self.scfg.MAX_TRADES_PER_DAY):
                        log(f"{symbol}: пропуск — дневной лимит сделок {self.scfg.MAX_TRADES_PER_DAY} достигнут", self.cfg.VERBOSE)
                        continue

                if self.scfg.COOLDOWN_BARS and self.scfg.COOLDOWN_BARS > 0:
                    cooldown_bars = int(self.scfg.COOLDOWN_BARS)
                    tf_seconds = self._timeframe_seconds()
                    last_exit_ts = self.last_exit_bar_ts.get(symbol)
                    if self._bar_ts is not None and last_exit_ts is not None and tf_seconds and tf_seconds > 0:
                        elapsed = (self._bar_ts - last_exit_ts).total_seconds()
                        bars_since = int(elapsed // tf_seconds) if elapsed > 0 else 0
                        if bars_since < cooldown_bars:
                            log(f"{symbol}: пропуск — кулдаун {self.scfg.COOLDOWN_BARS} баров после выхода", self.cfg.VERBOSE)
                            continue
                    else:
                        last_exit_idx = self.last_exit_bar_index.get(symbol)
                        if last_exit_idx is not None and (self._bar_index - int(last_exit_idx)) < cooldown_bars:
                            log(f"{symbol}: пропуск — кулдаун {self.scfg.COOLDOWN_BARS} баров после выхода", self.cfg.VERBOSE)
                            continue

                if self.losses_in_row >= self.scfg.MAX_LOSSES_IN_ROW:
                    log(f"{symbol}: пропуск (серия убыточных {self.losses_in_row} >= {self.scfg.MAX_LOSSES_IN_ROW})", self.cfg.VERBOSE)
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
                        quote_ccy = symbol.split('/')[1]
                        plan_profit = None
                        if pos.tp is not None:
                            plan_profit = max(0.0, (pos.tp - pos.entry) * pos.qty)
                        plan_loss = max(0.0, (pos.entry - pos.stop) * pos.qty)
                        plan_profit_str = f"{plan_profit:.4f} {quote_ccy}" if plan_profit is not None else "нет"
                        plan_loss_str = f"{plan_loss:.4f} {quote_ccy}"
                        log_trade(
                            f"ВХОД {symbol} направление=лонг объём={pos.qty:.8f} вход={fmt_float(pos.entry, 8)} "
                            f"стоп={fmt_float(pos.stop, 8)} тейк={fmt_float(pos.tp, 8) if pos.tp else 'нет'} "
                            f"план_прибыль={plan_profit_str} план_убыток={plan_loss_str}"
                        )
                        self.trades_today += 1
                    else:
                        log(f"{symbol}: не удалось выставить ордер", True)
                elif self.cfg.VERBOSE_CTX:
                    log(f"{symbol}: входа нет — ctx=" + format_ctx_str(ctx, 8))

            except (RequestTimeout, NetworkError, ExchangeNotAvailable, DDoSProtection) as e:
                log_error(f"Сеть/таймаут при обработке {symbol} в on_tick | исключение={e}")
                continue
            except Exception as e:
                log_error(f'Не удалось обработать {symbol} в on_tick', e)
                continue


class VolatilityDailyStrategy(BreakoutWithATRAndRSI):
    def _estimate_day_profit(self, qty: float, atr_val: float) -> Optional[float]:
        if qty <= 0 or atr_val <= 0:
            return None
        tf_seconds = self._timeframe_seconds()
        if not tf_seconds or tf_seconds <= 0:
            return None
        base_ts = self._bar_ts or datetime.utcnow()
        end_day = base_ts.replace(hour=23, minute=59, second=59, microsecond=0)
        if base_ts > end_day:
            end_day = end_day + timedelta(days=1)
        remaining_sec = max(0.0, (end_day - base_ts).total_seconds())
        remaining_bars = max(1, int(remaining_sec // tf_seconds))
        return atr_val * remaining_bars * qty

    def _signal(self, symbol: str, tf_df: pd.DataFrame, htf_df: Optional[pd.DataFrame]) -> Tuple[bool, Dict]:
        df = tf_df.copy()
        df['sma20'] = sma(df['close'], 20)
        df['atr'] = atr(df, 14)
        df['atr_pct'] = df['atr'] / df['close']

        last = df.iloc[-1]
        atr_ok = self.scfg.ATR_PCT_MIN <= float(last['atr_pct']) <= self.scfg.ATR_PCT_MAX
        trend_ok = float(last['close']) >= float(last['sma20'])
        will_long = bool(atr_ok and trend_ok)
        ctx = {
            'last_close': float(last['close']),
            'atr': float(last['atr']),
            'atr_pct': float(last['atr_pct']),
            'atr_ok': atr_ok,
            'trend_ok': trend_ok,
        }
        return will_long, ctx

    def _place_orders(self, symbol: str, qty: float, entry: float, atr_val: float, tf_df: pd.DataFrame) -> Optional[Position]:
        if qty <= 0 or entry <= 0:
            return None

        loss_per_unit = 1.0 / max(qty, 1e-12)
        desired_stop = entry - loss_per_unit
        if desired_stop <= 0:
            log(f"{symbol}: пропуск — стоп <= 0 при расчёте лимита убытка 1€", self.cfg.VERBOSE)
            return None

        minp = self.ex.min_price(symbol)
        stop_floor = max(minp, entry * (1.0 - self.scfg.STOP_MAX_PCT))
        stop = max(desired_stop, stop_floor)

        est_profit = self._estimate_day_profit(qty, atr_val)
        if est_profit is not None and est_profit < 1.0:
            log(f"{symbol}: пропуск — потенц. прибыль за день {est_profit:.4f} < 1€", self.cfg.VERBOSE)
            return None

        tp = entry + loss_per_unit if self.scfg.USE_TP else None

        if self.cfg.MODE == 'paper':
            return Position(symbol, 'long', qty, entry, stop, tp)

        buy = self.ex.create_market_buy(symbol, qty)
        filled_qty = float(buy.get('filled') or qty)
        filled_price = float(buy.get('average', buy.get('price', entry)) or entry)

        loss_per_unit = 1.0 / max(filled_qty, 1e-12)
        desired_stop = filled_price - loss_per_unit
        stop_floor = max(minp, filled_price * (1.0 - self.scfg.STOP_MAX_PCT))
        stop = max(desired_stop, stop_floor)
        tp = filled_price + loss_per_unit if self.scfg.USE_TP else None

        return Position(symbol, 'long', filled_qty, filled_price, stop, tp)


class DetailsStrategy:
    def __init__(self, cfg: Config, scfg: StrategyConfig, ex: Exchange):
        self.cfg = cfg
        self.scfg = scfg
        self.ex = ex
        self._logged = False

    def bootstrap_existing_positions(self):
        if not self._logged:
            log("Стратегия details пока не реализована — торговля не выполняется.", True)
            self._logged = True

    def on_tick(self):
        if not self._logged:
            log("Стратегия details пока не реализована — торговля не выполняется.", True)
            self._logged = True


STRATEGY_REGISTRY = {
    "first": BreakoutWithATRAndRSI,
    "details": DetailsStrategy,
    "second": VolatilityDailyStrategy,
}
