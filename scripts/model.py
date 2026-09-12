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


def novig(*odds):
    """Probabilità implicite senza margine (normalizzazione proporzionale)."""
    inv = np.array([1.0 / o for o in odds])
    return inv / inv.sum()
