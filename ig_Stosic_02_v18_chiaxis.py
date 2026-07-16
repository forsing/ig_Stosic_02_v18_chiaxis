from __future__ import annotations

# IG = Information Geometry (informaciona geometrija) 

"""
inspiration / upgrade  <--->  inspiracija / nadogradnja


Dragan Stošić / dva rada LUCES / ESP32 osvetljenje: 

1. Empirijska IG: Fisher metric, Multi-Chart (kad signal padne prelaz chartova), Christoffel / Levi-Civita, Histerezis.
https://zenodo.org/records/20094759
(DOI 10.5281/zenodo.20094759) — Fisher, chartovi, Christoffel, histerezis.

2. Ceo experimentalni sloj (paper + data + PVS) — ovo je „journal-ready“ paket. 
isti Manifold + mikro-ekscitacija + Fisher-preconditioned kontrola (A/B −25% jitter) + PVS dokazi + senzorski CSV.
https://zenodo.org/records/20389804
(novija PDF verzija: https://zenodo.org/records/20393695)
Naslov: Excitation-Dependent Observability Geometry…
Sadrži: paper 15 str, 6 CSV (boot…), serial logovi, PVS dokazi, A/B Boot 291 (GEO −25% jitter).
"""


"""
Fisher metrika na porodici raspodela nad istorijom (npr. frekvencije / uslovne raspodele)
multi-chart kad „observabilnost“ padne (npr. drugačiji režim / era)
natural gradient (Fisher precondition) ako nešto optimizujem 
histerezis putanja kroz vreme
mikro-ekscitacija (loto ne možeš da „probudiš“ kao lampu); PVS dokazi.
"""



"""
eksplicitna χ spektralna osa (paper 1).

Paper: χ = log((cold+ε)/(warm+ε)); U2 = (θ, χ) kad ρ padne.

Analog na 7/39 (po kolu):
  cold = # brojeva ≤13, warm = # ≥27
  χ_t = log((cold+ε)/(warm+ε))
θ_t = mean(draw) / 39   (skalarni „elevation“ proxy)

Fisher na putanji (χ): g_χχ ≈ 1 / var_rolling(χ)  (skalarna težina)
Po broju: skor = w_χ · doprinos broja χ-smeru + blagi excess
  doprinos: low → +χ, high → −χ (spram znaka cilja ka mean χ neighbors)

Ban last; next. CSV ceo, seed=39.
"""



import csv
from collections import Counter
from math import log
from pathlib import Path

import numpy as np

SEED = 39
FRONT_N = 39
FRONT_SELECT = 7
EPS = 1e-6
WINDOW = 100
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "loto7_4650_k56.csv"

np.random.seed(SEED)


def load_draws(csv_path: Path = CSV_PATH) -> np.ndarray:
    draws = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < FRONT_SELECT:
                continue
            try:
                draw = sorted(int(x.strip()) for x in row[:FRONT_SELECT])
            except ValueError:
                continue
            if len(draw) == FRONT_SELECT and all(1 <= x <= FRONT_N for x in draw):
                if len(set(draw)) == FRONT_SELECT:
                    draws.append(draw)
    if not draws:
        raise ValueError(f"Nema validnih kola u {csv_path}")
    return np.array(draws, dtype=int)


def chi_of(draw) -> float:
    cold = sum(1 for x in draw if int(x) <= 13)
    warm = sum(1 for x in draw if int(x) >= 27)
    return log((cold + EPS) / (warm + EPS))


def theta_of(draw) -> float:
    return float(np.mean(draw)) / float(FRONT_N)


def global_p(draws: np.ndarray) -> np.ndarray:
    cnt = Counter(draws.reshape(-1).tolist())
    n_slots = len(draws) * FRONT_SELECT
    return np.array([cnt.get(i, 0) / n_slots for i in range(1, FRONT_N + 1)], dtype=float)


def number_scores(
    draws: np.ndarray,
    chi_series: np.ndarray,
    g_chi: float,
    p_glob: np.ndarray,
    ban: set[int],
) -> dict[int, float]:
    """Cilj: χ next ≈ mean χ last WINDOW; brojevi pomeraju χ."""
    target = float(chi_series[-WINDOW:].mean()) if len(chi_series) >= 1 else 0.0
    chi_now = float(chi_series[-1])
    # smer: ako chi_now < target → treba više cold; obrnuto
    need_cold = target - chi_now
    out = {}
    for n in range(1, FRONT_N + 1):
        if n in ban:
            out[n] = -1e18
            continue
        if n <= 13:
            dchi = log((1 + EPS) / EPS)  # relative cold push
            align = need_cold
        elif n >= 27:
            dchi = -log((1 + EPS) / EPS)
            align = need_cold
        else:
            dchi = 0.0
            align = 0.0
        # Fisher weight on χ axis
        s = float(g_chi) * float(align) * float(dchi)
        s += 0.25 * float(p_glob[n - 1] - (1.0 / FRONT_N))
        out[n] = s
    return out


def _combo_fit(combo, score, target_sum, pos_means, target_odd, ban):
    nums = sorted(combo)
    if any(x in ban for x in nums):
        return -1e18
    s = sum(score[x] for x in nums)
    s -= 0.08 * abs(sum(nums) - target_sum)
    s -= 0.04 * sum(abs(nums[i] - pos_means[i]) for i in range(FRONT_SELECT))
    odd = sum(1 for x in nums if x % 2)
    s -= 0.3 * abs(odd - target_odd)
    return s


def predict_next(draws, score, ban):
    ranked = sorted((n for n in score if n not in ban), key=lambda n: (-score[n], n))
    target_sum = float(draws.sum(axis=1).mean())
    pos_means = [float(draws[:, i].mean()) for i in range(FRONT_SELECT)]
    target_odd = float(np.mean([sum(1 for x in d if x % 2) for d in draws]))
    candidates = [sorted(ranked[:FRONT_SELECT])]
    for start in range(0, min(20, len(ranked) - FRONT_SELECT + 1)):
        candidates.append(sorted(ranked[start : start + FRONT_SELECT]))
    best, best_fit = None, -1e18
    for base in candidates:
        fit = _combo_fit(base, score, target_sum, pos_means, target_odd, ban)
        if fit > best_fit:
            best_fit, best = fit, list(base)
        for i in range(FRONT_SELECT):
            for repl in ranked[:30]:
                cand = sorted(set(base[:i] + base[i + 1 :] + [repl]))
                if len(cand) != FRONT_SELECT:
                    continue
                fit = _combo_fit(cand, score, target_sum, pos_means, target_odd, ban)
                if fit > best_fit:
                    best_fit, best = fit, cand
    return best if best is not None else sorted(ranked[:FRONT_SELECT])


def run_ig_02_v18(csv_path: Path = CSV_PATH) -> None:
    draws = load_draws(csv_path)
    last = draws[-1]
    ban = set(int(x) for x in last.tolist())
    chi_series = np.array([chi_of(d) for d in draws], dtype=float)
    theta_series = np.array([theta_of(d) for d in draws], dtype=float)
    w = chi_series[-WINDOW:]
    var = float(np.var(w)) if len(w) > 1 else 1.0
    g_chi = 1.0 / (var + 1e-12)
    p_glob = global_p(draws)
    score = number_scores(draws, chi_series, g_chi, p_glob, ban)
    combo = predict_next(draws, score, ban)

    print(f"CSV: {csv_path.name}")
    print(f"Kola: {len(draws)} | seed={SEED} | WINDOW={WINDOW} | ig_02_v18 χ axis")
    print(f"last: {last.tolist()}")
    print()
    print("=== χ / θ ===")
    print(
        {
            "chi_now": round(float(chi_series[-1]), 6),
            "chi_mean_W": round(float(w.mean()), 6),
            "theta_now": round(float(theta_series[-1]), 6),
            "g_chi": round(g_chi, 4),
            "var_chi_W": round(var, 6),
        }
    )
    print()
    ranked = sorted(
        ((n, float(score[n])) for n in range(1, FRONT_N + 1) if n not in ban),
        key=lambda t: (-t[1], t[0]),
    )
    print("=== top12 skor ===")
    print([(n, round(sc, 6)) for n, sc in ranked[:12]])
    print()
    print("=== next (ig_02_v18 χ) ===")
    print("next:", combo)


if __name__ == "__main__":
    run_ig_02_v18()



"""
CSV: loto7_4650_k56.csv
Kola: 4650 | seed=39 | WINDOW=100 | ig_02_v18 χ axis
last: [4, 5, 6, 11, 12, 18, 28]

=== χ / θ ===
{'chi_now': 1.609437, 'chi_mean_W': 0.284657, 'theta_now': 0.307692, 'g_chi': 0.0698, 'var_chi_W': 14.321843}

=== top12 skor ===
[(34, 1.278262), (37, 1.278162), (32, 1.278139), (33, 1.278109), (29, 1.278093), (39, 1.27807), (35, 1.278024), (38, 1.278024), (31, 1.27794), (27, 1.277617), (30, 1.277594), (36, 1.277579)]

=== next (ig_02_v18 χ) ===
next: [14, x, 16, y, 21, z, 31]
"""
