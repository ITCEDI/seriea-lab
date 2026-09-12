"""
Test finale: Bet365 (il nostro book operativo) contro il consenso di mercato
(media di quote di 5-10 bookmaker tracciati da football-data.co.uk), non
contro un singolo book come Pinnacle. Segnala valore quando Bet365 paga
sistematicamente di piu' del consenso.

Aggiunge intervallo di confidenza bootstrap sul ROI, non solo il punto stimato:
un edge che sparisce nell'intervallo non e' un edge, e' rumore.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

NEED = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
        "B365H", "B365D", "B365A", "AvgH", "AvgD", "AvgA"]
OU_NEED = ["B365>2.5", "B365<2.5", "Avg>2.5", "Avg<2.5"]


def novig(*odds):
    inv = np.array([1 / o for o in odds])
    return inv / inv.sum()


def load():
    frames = []
    for f in sorted(glob.glob("data/history/*.csv")):
        d = pd.read_csv(f, encoding="latin-1")
        if not all(c in d.columns for c in NEED):
            continue
        cols = NEED + [c for c in OU_NEED if c in d.columns]
        d = d[cols].dropna(subset=NEED)
        frames.append(d)
    m = pd.concat(frames, ignore_index=True)
    m["Date"] = pd.to_datetime(m["Date"], format="mixed", dayfirst=True)
    return m.sort_values("Date").reset_index(drop=True)


def build_rows(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in m.iterrows():
        fh, fd, fa = novig(r.AvgH, r.AvgD, r.AvgA)
        for k, odds, fair, res in [
            ("H", r.B365H, fh, r.FTHG > r.FTAG),
            ("D", r.B365D, fd, r.FTHG == r.FTAG),
            ("A", r.B365A, fa, r.FTHG < r.FTAG),
        ]:
            rows.append({"date": r.Date, "market": f"1X2_{k}", "fair_p": fair,
                         "odds": odds, "edge": fair * odds - 1, "won": bool(res)})
        if pd.notna(r.get("Avg>2.5")) and pd.notna(r.get("B365>2.5")):
            fo, fu = novig(r["Avg>2.5"], r["Avg<2.5"])
            tot = r.FTHG + r.FTAG
            rows.append({"date": r.Date, "market": "OU_O25", "fair_p": fo,
                         "odds": r["B365>2.5"], "edge": fo * r["B365>2.5"] - 1, "won": tot > 2.5})
            rows.append({"date": r.Date, "market": "OU_U25", "fair_p": fu,
                         "odds": r["B365<2.5"], "edge": fu * r["B365<2.5"] - 1, "won": tot <= 2.5})
    return pd.DataFrame(rows)


def flat_roi(bets: pd.DataFrame, stake_frac=0.02, bk0=150.0) -> float:
    bk = bk0
    for _, r in bets.sort_values("date").iterrows():
        stake = bk * stake_frac
        bk += stake * (r.odds - 1) if r.won else -stake
    return (bk - bk0) / bk0


def bootstrap_ci(bets: pd.DataFrame, n_boot=3000, seed=42):
    """Ricampiona le giocate (non l'ordine temporale) per stimare l'incertezza
    del rendimento per-scommessa: media di (esito*payout) per singola giocata,
    che e' l'unita' statisticamente indipendente qui (non il bankroll composto,
    che introduce dipendenza dall'ordine)."""
    rng = np.random.default_rng(seed)
    per_bet_return = np.where(bets.won, bets.odds - 1, -1) * 0.02  # ritorno come frazione del bankroll corrente, approssimato flat
    n = len(per_bet_return)
    means = rng.choice(per_bet_return, size=(n_boot, n), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 50, 97.5])


def main():
    m = load()
    print(f"{len(m)} partite con Bet365 + media di mercato, dal {m.Date.min().date()} al {m.Date.max().date()}")
    all_rows = build_rows(m)
    all_rows.to_csv("data/clv_consensus_all.csv", index=False)

    print(f"\n{'soglia':>7} {'n':>6} {'win%':>7} {'p_fair':>8} {'odds':>7} {'ROI flat':>9} {'IC95% ritorno/bet':>22}")
    for th in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]:
        sel = all_rows[all_rows.edge >= th]
        if len(sel) < 15:
            print(f"{th:7.2f} {len(sel):6d}  n troppo basso")
            continue
        roi = flat_roi(sel)
        lo, mid, hi = bootstrap_ci(sel)
        sig = "***" if lo > 0 else ("neg" if hi < 0 else "n.s.")
        print(f"{th:7.2f} {len(sel):6d} {sel.won.mean()*100:6.1f}% {sel.fair_p.mean():8.3f} "
              f"{sel.odds.mean():7.2f} {roi*100:8.1f}%   [{lo*100:+.2f}%, {hi*100:+.2f}%] {sig}")

    print("\nPer mercato (soglia edge>=0.02):")
    sel = all_rows[all_rows.edge >= 0.02]
    print(sel.groupby("market").agg(n=("won", "size"), win_rate=("won", "mean"),
                                     avg_odds=("odds", "mean"), avg_edge=("edge", "mean")))


if __name__ == "__main__":
    main()
