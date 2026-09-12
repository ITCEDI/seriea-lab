# seriea-lab

Banco di prova (modalità carta) per un modello statistico sulla Serie A.

- `scripts/fetch_data.py` — raccolta dati: storico da football-data.co.uk, quote pre-partita da The Odds API (Bet365, Pinnacle; mercati 1X2 e Over/Under 2.5).
- `.github/workflows/fetch.yml` — esecuzione giornaliera automatica, con commit in `data/`.
- `data/history/` — un CSV per stagione, 2010/11 → corrente.
- `data/odds/latest.csv` — ultimo snapshot quote; `snapshots.csv` — tutti gli snapshot (per il closing line value).

Modello, backtest e cruscotto sono gestiti da una sessione Claude pianificata che legge da questo repository.
