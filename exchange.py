# -*- coding: utf-8 -*-
# Работа с биржей через ccxt и вспомогательные проверки.
from datetime import datetime, timezone
import time
from typing import List, Optional, Tuple

import ccxt
from ccxt.base.errors import DDoSProtection, ExchangeNotAvailable, NetworkError, RequestTimeout, InvalidOrder
import pandas as pd

from trading_2026.config import Config
from trading_2026.logging_utils import log, log_error


class Exchange:
    def __init__(self, cfg: Config):
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
        self.markets = self._call_with_retries("load_markets", self.ccxt.load_markets)
        self.has_oco = bool(getattr(self.ccxt, 'has', {}).get('createOrderOCO', False))
        self._avg_buy_price_logged = set()

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

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        ohlcv = self._call_with_retries(
            f"fetch_ohlcv({symbol},{timeframe},{limit})",
            self.ccxt.fetch_ohlcv,
            symbol,
            timeframe=timeframe,
            limit=limit,
        )
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df

    def fetch_balance(self) -> dict:
        return self._call_with_retries("fetch_balance", self.ccxt.fetch_balance)

    def balance_total_in_quote(self, quote: str = 'EUR') -> float:
        def _pair_last(base: str, q: str) -> Optional[float]:
            pair = f"{base}/{q}"
            try:
                t = self.ccxt.fetch_ticker(pair)
                px = float(t.get('last') or t.get('close') or ((t.get('bid') or 0) + (t.get('ask') or 0)) / 2 or 0.0)
                if px > 0:
                    return px
            except Exception:
                pass
            pair_rev = f"{q}/{base}"
            try:
                t = self.ccxt.fetch_ticker(pair_rev)
                px = float(t.get('last') or t.get('close') or ((t.get('bid') or 0) + (t.get('ask') or 0)) / 2 or 0.0)
                if px > 0:
                    return 1.0 / px if px != 0 else None
            except Exception:
                pass
            return None

        def _asset_in_quote(asset: str, q: str) -> Optional[float]:
            if asset == q:
                return 1.0
            px = _pair_last(asset, q)
            if px is not None:
                return px
            via = 'USDT'
            px1 = _pair_last(asset, via)
            px2 = _pair_last(via, q)
            if (px1 is not None) and (px2 is not None):
                return px1 * px2
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
                continue
            total_value += amount * price_q
        return float(total_value)

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

    def value_in_btc(self, asset: str, amount: float) -> Optional[float]:
        px = self.price_in_btc(asset)
        if px is None:
            return None
        try:
            return float(amount) * float(px)
        except Exception:
            return None

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

    def quote_free(self, quote: str = 'EUR') -> float:
        bal = self.ccxt.fetch_balance()
        try:
            return float(bal.get('free', {}).get(quote, 0.0) or 0.0)
        except Exception:
            return 0.0

    def max_buy_qty(self, symbol: str, safety: float = 0.995) -> float:
        quote = symbol.split('/')[1]
        free_quote = self.quote_free(quote)
        px = self.last_price(symbol)
        if px <= 0 or free_quote <= 0:
            return 0.0
        raw = (free_quote / px) * max(0.0, min(1.0, safety))
        if raw <= 0:
            return 0.0
        try:
            return float(self.ccxt.amount_to_precision(symbol, raw))
        except Exception:
            log(f"{symbol}: amount_to_precision для max_buy_qty отклонил расчётный объём={raw:.12g} — вернём 0", True)
            return 0.0

    def base_free(self, symbol: str) -> float:
        base = symbol.split('/')[0]
        try:
            bal = self.fetch_balance()
        except Exception:
            return 0.0
        try:
            return float(bal.get('free', {}).get(base, 0.0) or 0.0)
        except Exception:
            return 0.0

    def avg_buy_price(self, symbol: str, lookback_days: int = 180) -> Optional[float]:
        try:
            since_ms = int((datetime.now(timezone.utc).timestamp() - lookback_days * 86400) * 1000)
            trades = self.ccxt.fetch_my_trades(symbol, since=since_ms)
            buys = [t for t in trades if str(t.get('side')) == 'buy']
            if not buys:
                buys = []
            cost = 0.0
            amount = 0.0
            if buys:
                for t in buys:
                    px = float(t.get('price') or 0.0)
                    qty = float(t.get('amount') or 0.0)
                    if px > 0 and qty > 0:
                        cost += px * qty
                        amount += qty
            if amount > 0:
                avg = cost / amount
                if symbol not in self._avg_buy_price_logged:
                    log(f"{symbol}: avg_buy_price из trades за {lookback_days}d = {avg:.8f}", True)
                    self._avg_buy_price_logged.add(symbol)
                return avg

            orders = self.ccxt.fetch_orders(symbol, since=since_ms)
            for o in orders or []:
                if str(o.get('side')) != 'buy':
                    continue
                status = str(o.get('status') or '').lower()
                if status and status not in ('closed', 'filled'):
                    continue
                filled = float(o.get('filled') or 0.0)
                if filled <= 0:
                    continue
                avg_px = float(o.get('average') or 0.0)
                cost_val = float(o.get('cost') or 0.0)
                if avg_px > 0:
                    cost += avg_px * filled
                    amount += filled
                elif cost_val > 0:
                    cost += cost_val
                    amount += filled
            if amount <= 0:
                return None
            avg = cost / amount
            if symbol not in self._avg_buy_price_logged:
                log(f"{symbol}: avg_buy_price из orders за {lookback_days}d = {avg:.8f}", True)
                self._avg_buy_price_logged.add(symbol)
            return avg
        except Exception:
            return None

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[dict]:
        try:
            return self._call_with_retries("fetch_open_orders", self.ccxt.fetch_open_orders, symbol)
        except Exception:
            return []

    def market_info(self, symbol: str) -> dict:
        return self.markets[symbol]

    def last_price(self, symbol: str) -> float:
        t = self._call_with_retries("fetch_ticker", self.ccxt.fetch_ticker, symbol)
        return float(t.get('last') or t.get('close') or t.get('bid') or 0.0)

    def min_order_cost_quote(self, symbol: str, fallback_price: Optional[float] = None) -> Optional[float]:
        m = self.market_info(symbol)
        cost_limits = m.get('limits', {}).get('cost') or {}
        min_cost = cost_limits.get('min')
        if min_cost is not None:
            try:
                return float(min_cost)
            except Exception:
                pass
        amt_limits = m.get('limits', {}).get('amount') or {}
        min_amount = amt_limits.get('min')
        if min_amount is None:
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

    def price_step(self, symbol: str) -> float:
        m = self.market_info(symbol)
        try:
            for f in m.get('info', {}).get('filters', []):
                if f.get('filterType') == 'PRICE_FILTER':
                    ts = float(f.get('tickSize') or 0)
                    if ts > 0:
                        return ts
        except Exception:
            pass
        prec = (m.get('precision') or {}).get('price')
        if isinstance(prec, int) and prec >= 0:
            try:
                return float(10 ** (-prec))
            except Exception:
                pass
        lim_min = (m.get('limits', {}).get('price') or {}).get('min')
        try:
            return float(lim_min or 0.0)
        except Exception:
            return 0.0

    def min_price(self, symbol: str) -> float:
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

    def _percent_price_bounds(self, symbol: str, side: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        side_l = str(side).lower()
        m = self.market_info(symbol)
        filters = (m.get('info') or {}).get('filters', []) or []

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
            try:
                ref_price = self.last_price(symbol)
            except Exception:
                ref_price = None

        if (ref_price is None) or (ref_price <= 0) or (mult_down is None):
            return None, None, ref_price

        min_allowed = ref_price * mult_down
        max_allowed = (ref_price * mult_up) if (mult_up is not None) else None
        return min_allowed, max_allowed, ref_price

    def affordable(self, symbol: str, quote_balance: float) -> Tuple[bool, Optional[float]]:
        min_cost = self.min_order_cost_quote(symbol)
        if min_cost is None:
            return False, None
        return (quote_balance >= min_cost), float(min_cost)

    def round_qty(self, symbol: str, qty: float) -> float:
        info = self.market_info(symbol)
        limits_amount = (info.get('limits') or {}).get('amount') or {}
        min_amount = limits_amount.get('min')
        prec = (info.get('precision') or {}).get('amount')

        try:
            if isinstance(prec, int) and prec == 0 and qty < 1:
                log(f"{symbol}: объём {qty:.8f} < 1 при precision=0 (нужны целые значения) — пропускаем", True)
                return 0.0
        except Exception:
            pass

        try:
            qty_rounded = float(self.ccxt.amount_to_precision(symbol, qty))
        except Exception:
            log(f"{symbol}: amount_to_precision отклонил объём={qty:.12g} — пропускаем", True)
            return 0.0

        if min_amount is not None:
            try:
                min_amount = float(min_amount)
            except Exception:
                min_amount = None
        if (min_amount is not None) and (qty_rounded < min_amount):
            log(f"{symbol}: округлённый объём {qty_rounded:.8f} < минимального {min_amount:.8f} — пропускаем", True)
            return 0.0

        return float(qty_rounded)

    def create_market_buy(self, symbol: str, amount: float) -> dict:
        return self.ccxt.create_order(symbol, 'market', 'buy', amount)

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

    def create_limit_sell(self, symbol: str, amount: float, price: float, params: dict = None) -> dict:
        min_allowed, max_allowed, ref_price = self._percent_price_bounds(symbol, 'sell')
        if (min_allowed is not None) and (ref_price is not None) and ref_price > 0:
            try:
                price = max(price, min_allowed)
                if max_allowed is not None:
                    price = min(price, max_allowed)
            except Exception:
                pass

        amount_p = float(self.ccxt.amount_to_precision(symbol, amount))
        price_p = float(self.ccxt.price_to_precision(symbol, price))
        return self.ccxt.create_order(symbol, 'limit', 'sell', amount_p, price_p, params or {})

    def create_stop_loss_limit(self, symbol: str, amount: float, stop_price: float, limit_price: float) -> dict:
        min_allowed, max_allowed, ref_price = self._percent_price_bounds(symbol, 'sell')
        if (min_allowed is not None) and (ref_price is not None) and ref_price > 0:
            try:
                stop_price = max(stop_price, min_allowed)
                limit_price = max(limit_price, min_allowed)
                if max_allowed is not None:
                    stop_price = min(stop_price, max_allowed)
                    limit_price = min(limit_price, max_allowed)
            except Exception:
                pass

        amount_p = float(self.ccxt.amount_to_precision(symbol, amount))
        stop_p = float(self.ccxt.price_to_precision(symbol, stop_price))
        limit_p = float(self.ccxt.price_to_precision(symbol, limit_price))
        params = {
            'stopPrice': stop_p,
            'timeInForce': 'GTC',
        }
        return self.ccxt.create_order(symbol, 'stop_loss_limit', 'sell', amount_p, limit_p, params)

    def create_oco_sell(self, symbol: str, amount: float, take_profit_price: float, stop_price: float, stop_limit_price: float) -> dict:
        min_allowed, max_allowed, ref_price = self._percent_price_bounds(symbol, 'sell')
        if (min_allowed is not None) and (ref_price is not None) and ref_price > 0:
            try:
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

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> dict:
        try:
            return self.ccxt.cancel_order(order_id, symbol)
        except Exception:
            return {}

    def cancel_all_orders(self, symbol: Optional[str] = None) -> None:
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


def filter_markets_by_affordability(ex: Exchange, markets: List[str], total_quote_balance: float, verbose: bool = True) -> List[str]:
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
