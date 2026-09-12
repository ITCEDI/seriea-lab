"""
Modello Dixon-Coles per la Serie A.

Poisson bivariata con correzione rho sui punteggi bassi, forza di attacco e
difesa per squadra, fattore campo unico, decadimento temporale esponenziale
(xi per giorno). Le squadre non presenti nel campione di stima (neopromosse
appena arrivate) ereditano la forza media delle tre squadre più deboli.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10


def _tau(x, y, lam, mu, rho):
    """Correzione Dixon-Coles per 0-0, 1-0, 0-1, 1-1."""
    t = np.ones_like(lam)
    m00 = (x == 0) & (y == 0)
    m10 = (x == 1) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m11 = (x == 1) & (y == 1)
    t = np.where(m00, 1 - lam * mu * rho, t)
    t = np.where(m10, 1 + mu * rho, t)
    t = np.where(m01, 1 + lam * rho, t)
    t = np.where(m11, 1 - rho, t)
    return t


class DixonColes:
    def __init__(self, xi: float = 0.0018):
        self.xi = xi
        self.teams: list[str] = []
        self.params: np.ndarray | None = None
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}
        self.home_adv = 0.25
        self.rho = -0.05

    # ------------------------------------------------------------------ fit
    def fit(self, matches: pd.DataFrame, ref_date: pd.Timestamp, warm: bool = True):
        """matches: colonne Date, HomeTeam, AwayTeam, FTHG, FTAG."""
        m = matches.dropna(subset=["FTHG", "FTAG"]).copy()
        days = (ref_date - m["Date"]).dt.days.values.astype(float)
        w = np.exp(-self.xi * days)
        teams = sorted(set(m["HomeTeam"]) | set(m["AwayTeam"]))
        idx = {t: i for i, t in enumerate(teams)}
        h = m["HomeTeam"].map(idx).values
        a = m["AwayTeam"].map(idx).values
        x = m["FTHG"].values.astype(int)
        y = m["FTAG"].values.astype(int)
        n = len(teams)

        # warm start: riusa i parametri delle squadre già note
        if warm and self.params is not None and self.teams:
            att0 = np.array([self.attack.get(t, 0.0) for t in teams])
            dfn0 = np.array([self.defence.get(t, 0.0) for t in teams])
            p0 = np.concatenate([att0, dfn0, [self.home_adv, self.rho]])
        else:
            p0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25, -0.05]])

        lgx = np.log(np.array([math.factorial(k) for k in range(MAX_GOALS + 3)]))

        def nll(p):
            att = p[:n] - p[:n].mean()  # vincolo somma zero
            dfn = p[n:2 * n]
            gamma, rho = p[-2], p[-1]
            lam = np.exp(att[h] + dfn[a] + gamma)
            mu = np.exp(att[a] + dfn[h])
            ll = (x * np.log(lam) - lam - lgx[x]) + (y * np.log(mu) - mu - lgx[y])
            t = _tau(x, y, lam, mu, rho)
            t = np.clip(t, 1e-6, None)
            ll += np.log(t)
            return -(w * ll).sum()

        bounds = [(-3, 3)] * (2 * n) + [(-1, 1), (-0.5, 0.5)]
        res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 300, "ftol": 1e-9})
        p = res.x
        att = p[:n] - p[:n].mean()
        dfn = p[n:2 * n]
        self.teams = teams
        self.params = p
        self.attack = {t: float(att[i]) for t, i in idx.items()}
        self.defence = {t: float(dfn[i]) for t, i in idx.items()}
        self.home_adv = float(p[-2])
        self.rho = float(p[-1])
        return self

    # -------------------------------------------------------------- predict
    def _strength(self, team: str):
        if team in self.attack:
            return self.attack[team], self.defence[team]
        # fallback: media delle tre squadre più deboli (attacco - difesa)
        ranking = sorted(self.teams, key=lambda t: self.attack[t] - self.defence[t])[:3]
        return (float(np.mean([self.attack[t] for t in ranking])),
                float(np.mean([self.defence[t] for t in ranking])))

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        ah, dh = self._strength(home)
        aa, da = self._strength(away)
        lam = np.exp(ah + da + self.home_adv)
        mu = np.exp(aa + dh)
        gh = poisson.pmf(np.arange(MAX_GOALS + 1), lam)
        ga = poisson.pmf(np.arange(MAX_GOALS + 1), mu)
        M = np.outer(gh, ga)
        # correzione rho
        M[0, 0] *= 1 - lam * mu * self.rho
        M[1, 0] *= 1 + mu * self.rho
        M[0, 1] *= 1 + lam * self.rho
        M[1, 1] *= 1 - self.rho
        M = np.clip(M, 0, None)
        return M / M.sum()

    def predict(self, home: str, away: str) -> dict:
        M = self.score_matrix(home, away)
        p_home = np.tril(M, -1).sum()
        p_draw = np.trace(M)
        p_away = np.triu(M, 1).sum()
        tot = np.add.outer(np.arange(MAX_GOALS + 1), np.arange(MAX_GOALS + 1))
        p_over = M[tot > 2.5].sum()
        ah, dh = self._strength(home)
        aa, da = self._strength(away)
        return {
            "pH": float(p_home), "pD": float(p_draw), "pA": float(p_away),
            "pO25": float(p_over), "pU25": float(1 - p_over),
            "xG_home": float(np.exp(ah + da + self.home_adv)),
            "xG_away": float(np.exp(aa + dh)),
            "known_home": home in self.attack, "known_away": away in self.attack,
        }


def predict_from_lambda(lam: float, mu: float, rho: float) -> dict:
    """Stessa logica di DixonColes.predict/score_matrix ma con lambda/mu gia'
    calcolati (usata per applicare aggiustamenti qualitativi a xG noti)."""
    gh = poisson.pmf(np.arange(MAX_GOALS + 1), lam)
    ga = poisson.pmf(np.arange(MAX_GOALS + 1), mu)
    M = np.outer(gh, ga)
    M[0, 0] *= 1 - lam * mu * rho
    M[1, 0] *= 1 + mu * rho
    M[0, 1] *= 1 + lam * rho
    M[1, 1] *= 1 - rho
    M = np.clip(M, 0, None)
    M = M / M.sum()
    p_home = np.tril(M, -1).sum()
    p_draw = np.trace(M)
    p_away = np.triu(M, 1).sum()
    tot = np.add.outer(np.arange(MAX_GOALS + 1), np.arange(MAX_GOALS + 1))
    p_over = M[tot > 2.5].sum()
    return {"pH": float(p_home), "pD": float(p_draw), "pA": float(p_away),
            "pO25": float(p_over), "pU25": float(1 - p_over),
            "xG_home": float(lam), "xG_away": float(mu)}


def market_probs(M: np.ndarray) -> dict:
    """Tutte le probabilita' derivabili dalla stessa matrice dei punteggi:
    1X2, doppia chance, Under/Over su piu' linee, Multigol, Gol/NoGol,
    handicap asiatico semplice, punteggio esatto piu' probabile."""
    n = M.shape[0]
    tot = np.add.outer(np.arange(n), np.arange(n))
    diff = np.subtract.outer(np.arange(n), np.arange(n))  # gol casa - gol trasferta

    pH = M[diff > 0].sum(); pD = M[diff == 0].sum(); pA = M[diff < 0].sum()

    out = {
        "1X2": {"1": float(pH), "X": float(pD), "2": float(pA)},
        "doppia_chance": {"1X": float(pH + pD), "X2": float(pD + pA), "12": float(pH + pA)},
        "under_over": {},
        "multigol": {},
        "gol_nogol": {
            "GG": float(M[(np.arange(n)[:, None] > 0) & (np.arange(n)[None, :] > 0)].sum()),
        },
        "handicap": {},
    }
    out["gol_nogol"]["NG"] = 1 - out["gol_nogol"]["GG"]

    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        over = float(M[tot > line].sum())
        out["under_over"][f"O{line}"] = over
        out["under_over"][f"U{line}"] = 1 - over

    mg_ranges = [(1, 3), (2, 4), (2, 5), (0, 1), (4, 6)]
    for lo, hi in mg_ranges:
        out["multigol"][f"{lo}-{hi}"] = float(M[(tot >= lo) & (tot <= hi)].sum())

    for h in [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]:
        # handicap asiatico sulla squadra di casa: vince se (gol_casa + h) > gol_trasferta
        adj = diff + h
        win = float(M[adj > 0].sum())
        push = float(M[adj == 0].sum()) if h in (-1.0, 1.0) else 0.0
        lose = 1 - win - push
        out["handicap"][f"casa {h:+.1f}"] = {"vince": win, "push": push, "perde": lose}

    top_idx = np.dstack(np.unravel_index(np.argsort(-M, axis=None)[:5], M.shape))[0]
    out["punteggi_esatti"] = [{"risultato": f"{i}-{j}", "p": float(M[i, j])} for i, j in top_idx]
    return out


def _candidates(mp: dict) -> list:
    candidates = []
    for k, v in mp["1X2"].items():
        candidates.append((f"1X2: {k}", v, 1))
    for k, v in mp["doppia_chance"].items():
        candidates.append((f"Doppia chance: {k}", v, 1))
    for k, v in mp["under_over"].items():
        if k in ("O2.5", "U2.5", "O1.5", "U1.5"):
            candidates.append((k, v, 1))
    candidates.append(("Gol (GG)", mp["gol_nogol"]["GG"], 2))
    candidates.append(("No Gol (NG)", mp["gol_nogol"]["NG"], 2))
    for k, v in mp["multigol"].items():
        candidates.append((f"Multigol {k}", v, 2))
    for k, v in mp["handicap"].items():
        candidates.append((f"Handicap {k}", v["vince"], 3))
    return candidates


def most_probable_pick(mp: dict) -> dict:
    """L'esito piu' probabile in assoluto tra i mercati principali, qualunque
    sia la sua quota (puo' essere un evento quasi scontato a quota bassa) —
    diverso da wisest_pick, che invece cerca una quota vicina a 2. Esclude gli
    handicap: tecnicamente spesso i piu' probabili in assoluto, ma poco
    leggibili come 'pronostico' nel senso comune del termine."""
    candidates = [c for c in _candidates(mp) if c[2] < 3]
    candidates.sort(key=lambda c: (-c[1], c[2]))
    best = candidates[0]
    return {"mercato": best[0], "probabilita": best[1],
            "quota_equa_stimata": (1 / best[1]) if best[1] > 0 else None}


def wisest_pick(mp: dict, odds_target: tuple = (1.8, 2.3)) -> dict:
    """Tra tutti i mercati derivabili dalla matrice, sceglie quello con la
    probabilita' del modello piu' vicina alla fascia di quota richiesta
    (default 1.80-2.30, cioe' 'circa quota 2'), preferendo mercati semplici
    e leggibili (1X2, doppia chance, O/U, GG/NG, multigol) alle combinazioni
    piu' esotiche (handicap, punteggio esatto), che restano come riserva."""
    lo_p, hi_p = 1 / odds_target[1], 1 / odds_target[0]  # banda di probabilita' equivalente
    candidates = _candidates(mp)
    in_band = [c for c in candidates if lo_p <= c[1] <= hi_p]
    pool = in_band if in_band else candidates
    # tra quelli in fascia: probabilita' piu' alta prima, poi mercati piu' semplici (tier piu' basso)
    pool.sort(key=lambda c: (-c[1], c[2]))
    best = pool[0]
    quota_equa = 1 / best[1] if best[1] > 0 else None
    return {
        "mercato": best[0], "probabilita": best[1], "quota_equa_stimata": quota_equa,
        "in_fascia_richiesta": bool(in_band),
    }


def novig(*odds):
    """Probabilità implicite senza margine (normalizzazione proporzionale)."""
    inv = np.array([1.0 / o for o in odds])
    return inv / inv.sum()
