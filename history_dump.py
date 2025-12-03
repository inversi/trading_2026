# history_dump.py
# Скрипт для разовой выгрузки истории ордеров и сделок за последний месяц.

from bot import Exchange, load_config, collect_history_last_month


def main():
    # Загружаем конфиг так же, как это делает бот
    cfg = load_config()

    # Создаём обёртку биржи
    ex = Exchange(cfg)

    # Выгружаем историю по всем рынкам из cfg.MARKETS за последние 30 дней
    collect_history_last_month(ex, symbols=cfg.MARKETS, days=30)


if __name__ == "__main__":
    main()