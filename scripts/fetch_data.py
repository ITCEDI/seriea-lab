#!/usr/bin/env python3
"""
Raccolta dati per seriea-lab.

1. Storico Serie A da football-data.co.uk (risultati + quote di chiusura),
   stagioni 2010/11 -> corrente. Salva un CSV per stagione in data/history/.
2. Quote pre-partita delle prossime gare da The Odds API (Bet365 + Pinnacle,
   mercati 1X2 e Over/Under 2.5). Salva lo snapshot del giorno in data/odds/
   e lo accoda a data/odds/snapshots.csv (serve per il closing line value).

Eseguito da GitHub Actions; la chiave API arriva dalla variabile d'ambiente
ODDS_API_KEY. Non stampa mai la chiave.
"""
import csv
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
HIST_DIR = ROOT / "data" / "history"
ODDS_DIR = ROOT / "data" / "odds"
HIST_DIR.mkdir(parents=True, exist_ok=True)
ODDS_DIR.mkdir(parents=True, exist_ok=True)

FD_BASE = "https://www.football-data.co.uk/mmz4281/{season}/I1.csv"
FIRST_SEASON = 2010  # 2010/11


def current_season_start(today: datetime) -> int:
    # La stagione YYYY/YY+1 inizia ad agosto.
    return today.year if today.month >= 7 else today.year - 1


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def fetch_history(today: datetime) -> None:
    cur = current_season_start(today)
    for y in range(FIRST_SEASON, cur + 1):
        code = season_code(y)
        out = HIST_DIR / f"I1_{code}.csv"
        # Le stagioni concluse non cambiano: scarica una volta sola.
        if out.exists() and y < cur:
            continue
        url = FD_BASE.format(season=code)
        r = requests.get(url, timeout=60)
        if r.status_code != 200 or len(r.content) < 1000:
            print(f"[history] {code}: HTTP {r.status_code}, {len(r.content)} bytes -> skip")
            continue
        # football-data usa latin-1 e talvolta righe vuote in coda
        text = r.content.decode("latin-1")
        rows = [row for row in csv.reader(io.StringIO(text)) if any(c.strip() for c in row)]
        with open(out, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        print(f"[history] {code}: {len(rows) - 1} partite")


def fetch_odds(today: datetime) -> None:
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        print("[odds] ODDS_API_KEY assente: salto le quote")
        return
    url = "https://api.the-odds-api.com/v4/sports/soccer_italy_serie_a/odds"
    params = {
        "apiKey": key,
        "regions": "eu",
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
        "bookmakers": "bet365,pinnacle",
        "dateFormat": "iso",
    }
    r = requests.get(url, params=params, timeout=60)
    if r.status_code != 200:
        print(f"[odds] HTTP {r.status_code}: {r.text[:200]}")
        sys.exit(1)
    remaining = r.headers.get("x-requests-remaining")
    used = r.headers.get("x-requests-used")
    print(f"[odds] crediti usati {used}, rimanenti {remaining}")
    events = r.json()

    stamp = today.strftime("%Y-%m-%dT%H%M")
    raw_path = ODDS_DIR / f"raw_{today.strftime('%Y-%m-%d')}.json"
    raw_path.write_text(json.dumps(events, ensure_ascii=False, indent=1), encoding="utf-8")

    # Normalizzazione: una riga per evento/bookmaker/mercato/esito
    rows = []
    for ev in events:
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                for oc in mk.get("outcomes", []):
                    if mk["key"] == "totals" and oc.get("point") != 2.5:
                        continue
                    rows.append({
                        "snapshot": stamp,
                        "event_id": ev["id"],
                        "commence": ev["commence_time"],
                        "home": ev["home_team"],
                        "away": ev["away_team"],
                        "bookmaker": bk["key"],
                        "market": mk["key"],
                        "outcome": oc["name"],
                        "point": oc.get("point", ""),
                        "price": oc["price"],
                        "last_update": mk.get("last_update", ""),
                    })
    fields = list(rows[0].keys()) if rows else [
        "snapshot", "event_id", "commence", "home", "away", "bookmaker",
        "market", "outcome", "point", "price", "last_update"]
    latest = ODDS_DIR / "latest.csv"
    with open(latest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    snaps = ODDS_DIR / "snapshots.csv"
    new_file = not snaps.exists()
    with open(snaps, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        w.writerows(rows)
    print(f"[odds] {len(events)} eventi, {len(rows)} righe quote")


def main() -> None:
    today = datetime.now(timezone.utc)
    fetch_history(today)
    fetch_odds(today)
    (ROOT / "data" / "last_update.txt").write_text(today.isoformat(), encoding="utf-8")


if __name__ == "__main__":
    main()
