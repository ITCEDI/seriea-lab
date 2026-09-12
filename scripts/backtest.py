"""
Backtest walk-forward del modello Dixon-Coles su tutte le stagioni storiche.

Per ogni partita, il modello viene stimato SOLO con le partite precedenti
(mai con dati futuri). Si confrontano le probabilità stimate con le quote
Bet365 (book operativo) e si segnala "valore" quando p_modello * quota > 1 + soglia.

Confronto con "giocare a caso": N simulazioni di scommesse casuali sugli
stessi eventi e mercati, stesso staking, per collocare il rendimento reale
nella distribuzione del caso.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import DixonColes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "history"

# football-data.co.uk usa nomi leggermente diversi da The Odds API in alcuni casi;
# per ora il backtest storico lavora solo sui nomi football-data (coerenti tra loro).

REFIT_EVERY_DAYS = 7      # ristima settimanale, non partita per partita (costo)
MIN_TRAIN_MATCHES = 380   # almeno una stagione piena prima di iniziare a scommettere
EDGE_THRESHOLD = 0.03     # soglia minima di valore stimato
STAKE_FRACTION = 0.02     # flat staking
KELLY_FRACTION = 0.25
BANKROLL0 = 150.0
N_RANDOM_SIMS = 10000
RNG_SEED = 42


def load_all_seasons() -> pd.DataFrame:
    frames = []
    for f in sorted(HIST.glob("I1_*.csv")):
        d = pd.read_csv(f, encoding="latin-1")
        d["Date"] = pd.to_datetime(d["Date"], format="mixed", dayfirst=True, errors="coerce")
        keep = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "B365H", "B365D", "B365A"]
        for c in ["B365>2.5", "B365<2.5"]:
            if c in d.columns:
                keep.append(c)
        d = d[[c for c in keep if c in d.columns]].dropna(subset=["Date", "HomeTeam", "AwayTeam"])
        frames.append(d)
    all_m = pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
    all_m = all_m.dropna(subset=["FTHG", "FTAG"])
    return all_m


def implied_novig_1x2(row) -> tuple[float, float, float] | None:
    if pd.isna(row.get("B365H")) or pd.isna(row.get("B365D")) or pd.isna(row.get("B365A")):
        return None
    inv = np.array([1 / row["B365H"], 1 / row["B365D"], 1 / row["B365A"]])
    return tuple(inv / inv.sum())


def run_backtest(matches: pd.DataFrame) -> pd.DataFrame:
    model = DixonColes()
    bets = []
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
            continue  # non scommettiamo su squadre mai viste (prime giornate dopo promozione)

        outcomes = {
            "1X2_H": (pred["pH"], row.get("B365H"), row["FTHG"] > row["FTAG"]),
            "1X2_D": (pred["pD"], row.get("B365D"), row["FTHG"] == row["FTAG"]),
            "1X2_A": (pred["pA"], row.get("B365A"), row["FTHG"] < row["FTAG"]),
        }
        if "B365>2.5" in row.index and pd.notna(row.get("B365>2.5")):
            outcomes["OU_O25"] = (pred["pO25"], row["B365>2.5"], (row["FTHG"] + row["FTAG"]) > 2.5)
            outcomes["OU_U25"] = (pred["pU25"], row["B365<2.5"], (row["FTHG"] + row["FTAG"]) <= 2.5)

        for market, (p_model, odds, won) in outcomes.items():
            if odds is None or pd.isna(odds) or odds <= 1.0:
                continue
            edge = p_model * odds - 1
            if edge < EDGE_THRESHOLD:
                continue
            b = (odds - 1) * p_model - (1 - p_model)  # Kelly criterion numerator
            kelly = max(0.0, b / (odds - 1)) * KELLY_FRACTION
            bets.append({
                "date": date, "home": row["HomeTeam"], "away": row["AwayTeam"],
                "market": market, "p_model": p_model, "odds": odds, "edge": edge,
                "kelly_frac": kelly, "won": bool(won),
            })

    return pd.DataFrame(bets)


def apply_staking(bets: pd.DataFrame) -> pd.DataFrame:
    bets = bets.sort_values("date").reset_index(drop=True)
    bk_flat = BANKROLL0
    bk_kelly = BANKROLL0
    flat_curve, kelly_curve = [], []
    for _, r in bets.iterrows():
        stake_flat = bk_flat * STAKE_FRACTION
        pnl_flat = stake_flat * (r["odds"] - 1) if r["won"] else -stake_flat
        bk_flat += pnl_flat
        flat_curve.append(bk_flat)

        stake_kelly = bk_kelly * r["kelly_frac"]
        pnl_kelly = stake_kelly * (r["odds"] - 1) if r["won"] else -stake_kelly
        bk_kelly += pnl_kelly
        kelly_curve.append(bk_kelly)
    bets["bankroll_flat"] = flat_curve
    bets["bankroll_kelly"] = kelly_curve
    return bets


def random_baseline(bets: pd.DataFrame, n_sims=N_RANDOM_SIMS, seed=RNG_SEED) -> np.ndarray:
    """N simulazioni: stesso numero di scommesse, stesse quote, esito casuale
    pescato secondo la probabilità no-vig del bookmaker (non quella del modello) —
    cioè 'giocare a caso ma senza pregiudizio', il confronto più severo possibile."""
    rng = np.random.default_rng(seed)
    odds = bets["odds"].values
    p_fair = bets["p_model"].values  # placeholder, sovrascritto sotto se disponibile
    finals = np.empty(n_sims)
    for s in range(n_sims):
        bk = BANKROLL0
        for j in range(len(bets)):
            stake = bk * STAKE_FRACTION
            win = rng.random() < bets["p_fair_book"].iloc[j]
            bk += stake * (odds[j] - 1) if win else -stake
        finals[s] = bk
    return finals


def main():
    print("Carico storico...")
    matches = load_all_seasons()
    print(f"{len(matches)} partite totali, dal {matches['Date'].min().date()} al {matches['Date'].max().date()}")

    print("Eseguo backtest walk-forward (può richiedere qualche minuto)...")
    bets = run_backtest(matches)
    print(f"{len(bets)} giocate suggerite dal modello")

    if bets.empty:
        print("Nessuna giocata sopra soglia: verificare parametri.")
        return

    # probabilità no-vig del book per il confronto casuale onesto
    def fair_prob(row):
        m = matches[(matches["Date"] == row["date"]) & (matches["HomeTeam"] == row["home"]) & (matches["AwayTeam"] == row["away"])]
        if m.empty:
            return np.nan
        m = m.iloc[0]
        if row["market"].startswith("1X2"):
            probs = implied_novig_1x2(m)
            if probs is None:
                return np.nan
            return {"1X2_H": probs[0], "1X2_D": probs[1], "1X2_A": probs[2]}[row["market"]]
        else:
            o, u = m.get("B365>2.5"), m.get("B365<2.5")
            if pd.isna(o) or pd.isna(u):
                return np.nan
            inv = np.array([1 / o, 1 / u])
            fair = inv / inv.sum()
            return fair[0] if row["market"] == "OU_O25" else fair[1]

    bets["p_fair_book"] = bets.apply(fair_prob, axis=1)
    bets = bets.dropna(subset=["p_fair_book"])

    bets = apply_staking(bets)

    n = len(bets)
    win_rate = bets["won"].mean()
    total_staked_flat = (bets["bankroll_flat"].shift(1).fillna(BANKROLL0) * STAKE_FRACTION).sum()
    roi_flat = (bets["bankroll_flat"].iloc[-1] - BANKROLL0) / total_staked_flat
    yield_flat = (bets["bankroll_flat"].iloc[-1] - BANKROLL0) / BANKROLL0
    clv = (bets["p_model"] - bets["p_fair_book"]).mean()

    print("\n--- Risultati backtest ---")
    print(f"Giocate valide (con quota no-vig disponibile): {n}")
    print(f"% vinte: {win_rate:.1%}")
    print(f"Bankroll finale (flat 2%): {bets['bankroll_flat'].iloc[-1]:.2f} EUR (partenza {BANKROLL0})")
    print(f"Bankroll finale (Kelly 1/4): {bets['bankroll_kelly'].iloc[-1]:.2f} EUR")
    print(f"ROI (flat): {roi_flat:.1%}  |  Yield (flat): {yield_flat:.1%}")
    print(f"CLV medio (p_modello - p_fair_book): {clv:+.3f}  ({'positivo' if clv > 0 else 'negativo'})")

    print("\nSimulo 10.000 stagioni casuali per confronto...")
    finals = random_baseline(bets)
    pct = (finals < bets["bankroll_flat"].iloc[-1]).mean()
    print(f"Il risultato reale batte il {pct:.1%} delle simulazioni casuali")
    print(f"Random: media {finals.mean():.2f}, mediana {np.median(finals):.2f}, "
          f"p05={np.percentile(finals,5):.2f}, p95={np.percentile(finals,95):.2f}")

    out = ROOT / "data" / "backtest_bets.csv"
    bets.to_csv(out, index=False)
    print(f"\nDettaglio giocate salvato in {out}")


if __name__ == "__main__":
    main()
