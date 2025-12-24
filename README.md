# trading_2026

Минималистичный спот-бот для Binance на ccxt. Стратегия: пробой максимума предыдущей свечи с фильтрами ATR/RSI/объема.

## Структура проекта
- `trading_2026/bot.py` — точка входа, префлайт, цикл, heartbeat.
- `trading_2026/config.py` — конфиг и загрузка `.env`.
- `trading_2026/logging_utils.py` — логирование и форматтеры.
- `trading_2026/indicators.py` — EMA/SMA/RSI/ATR.
- `trading_2026/exchange.py` — работа с биржей (ccxt).
- `trading_2026/strategy.py` — стратегия, позиции, риск-менеджмент.
- `trading_2026/history.py` — выгрузка истории сделок/ордеров.

## Быстрый старт
1) Создай `.env` (см. примеры в комментариях/коде).
2) Установи зависимости:
   `pip3 install -r trading_2026/requirements.txt`
3) Запуск:
   `python3 trading_2026/bot.py`
   или
   `python3 -m trading_2026.bot`

## Деплой
1) Сделать скрипт исполняемым (один раз):
   `chmod +x trading_2026/deploy.sh`
2) Запуск деплоя:
   `./trading_2026/deploy.sh`
3) После правок:
   `git add .`
   `git commit -m "update"`
   `git push origin main`
   `./trading_2026/deploy.sh`

## Логи
- `logs/app.log` — общий поток.
- `logs/trades.log` — входы/выходы.
- `logs/errors.log` — ошибки.
- Heartbeat каждые 2 минуты: "Heartbeat: бот работает".

## Важные настройки
- В `.env` остались инфраструктурные параметры (ключи, рынки, режим, биржа).
- Параметры стратегии лежат в `trading_2026/strategies/*.yaml`.
- Переключение стратегии: `STRATEGY=first` или `STRATEGY=details` в `.env`.

## Стратегии
- `trading_2026/strategies/first.yaml` — текущая стратегия (Breakout).
- `trading_2026/strategies/details.yaml` — заготовка, пока не реализована.

## Примечания
- Dust-остатки ниже `minNotional` игнорируются, чтобы не блокировать новые входы.
- Рекомендуется сначала `MODE=paper`, потом `MODE=live`.
