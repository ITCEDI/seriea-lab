"""
Diagnostica di calibrazione: per OGNI partita (non solo quelle sopra soglia
di valore) registra la probabilità del modello walk-forward e l'esito reale,
per capire DOVE nasce il bias prima di decidere le regole di scommessa.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import DixonColes  # noqa: E402
from backtest import load_all_seasons, REFIT_EVERY_DAYS, MIN_TRAIN_MATCHES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    matches = load_all_seasons()
    model = DixonColes()
    rows = []
    last_fit_date = None

    for i in range(MIN_TRAIN_MATCHES, len(matches)):
        row = matches.iloc[i]
        date = row["Date"]
        train = matches.iloc[:i]
        if last_fit_date is None or (date - last_fit_date).days >= REFIT_EVERY_DAYS:
            model.fit(train, ref_date=date)
            last_fit_date = date
        pred = model.predict(row["HomeTeam"], row["AwayTeam"])
        if not (pred["known_home"] and pred["known_away"]):
            continue
        rows.append({
            "date": date, "home": row["HomeTeam"], "away": row["AwayTeam"],
            "pH": pred["pH"], "pD": pred["pD"], "pA": pred["pA"],
            "pO25": pred["pO25"], "pU25": pred["pU25"],
            "B365H": row.get("B365H"), "B365D": row.get("B365D"), "B365A": row.get("B365A"),
            "B365O25": row.get("B365>2.5"), "B365U25": row.get("B365<2.5"),
            "res_H": row["FTHG"] > row["FTAG"], "res_D": row["FTHG"] == row["FTAG"],
            "res_A": row["FTHG"] < row["FTAG"],
            "total": row["FTHG"] + row["FTAG"],
        })
        if len(rows) % 500 == 0:
            print(f"{len(rows)} partite processate...", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data" / "calibration_raw.csv", index=False)
    print(f"Salvato: {len(df)} partite in data/calibration_raw.csv")


if __name__ == "__main__":
    main()
