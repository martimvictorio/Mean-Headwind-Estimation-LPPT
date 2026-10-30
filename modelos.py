"""
Modelos de estimação do headwind na final: filtro de Kalman, baseline EUROCONTROL e
benchmark METAR. Módulo partilhado pela afinação (Cap. 4) e pelos resultados (Cap. 5).

Referências:
  [T]   Tese WindOnFinal: §3.3 (filtro), §3.4 (baseline), §3.5 (benchmark METAR).
  [S21] Sun, Marinescu, Olivares & Staffetti (2021), ATM Seminar, Sec. III-A e IV.
  [EC7] EUROCONTROL Guidelines on TBS with ORD (2024), via T §2.2.2.

Origem: AdaptedKalmanFilter e run_filter vêm sem alterações de 'kalman_tuning_1julho.py';
EurocontrolBaseline, exit_step, run_baseline e os parâmetros do baseline vêm sem alterações
de 'baseline_1julho.py' (versões de 22-10-2026). Novo: benchmark METAR e interface comum.

Interface comum (Model): estimate(obs_by_step, start_step, query_steps, exclude=None)
devolve {k: vetor de 36 estimativas em kt} para cada passo k pedido; exclude omite as
observações de uma passagem (leave-one-out da avaliação nowcast, T Eq. 3.36). É esta
interface que permite avaliar os três modelos com o mesmo código (avaliacao.py).
"""

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

import preprocessamento as pp

# =============================================================================
# PARÂMETROS DOS MODELOS (único sítio onde são definidos)
# =============================================================================
KALMAN = {
    "sigma_obs_kt": 5.83,     # σ = 3 m/s (T §3.3, de Jong et al. via S21); R0 = σ² (Eq. 3.29)
    # P0 = c·σ²·I com c = 10 (T §4.3.2). Desvio a S21 §IV, que usa P0 = σ⁴·I.
    "p0_c": 10.0,
    "gate_ms": 10.0,          # T Eq. (3.20): rejeita |inovação| > 10 m/s
    # q afinado em 1 de julho (T §4.3.2, Fig. 4.13). NÃO é reafinado nos cenários de avaliação.
    "q_kt2_s": 0.005,
}

BASELINE = {
    "window_s": 600,          # T Eq. (3.31): 10 min [EC7]
    "outlier_ms": 10.0,       # T Eq. (3.33): 10 m/s [EC7]
    "min_obs": 3,             # T Eq. (3.34): N ≥ 3 [EC7]
}


def kalman_derived(params):
    """R0, P0 e gate em kt derivados dos parâmetros do filtro. Único sítio onde são calculados."""
    r0 = params["sigma_obs_kt"] ** 2                          # T Eq. (3.29): R0 = σ²
    return {"r0_kt2": r0,
            "p0_kt2": params["p0_c"] * r0,                    # T §4.3.2: P0 = c·σ²
            "gate_kt": params["gate_ms"] * pp.MS_TO_KT}       # T Eq. (3.20): 10 m/s em kt


def baseline_params():
    """Parâmetros derivados. Limiar em kt pela mesma conversão usada no resto do código."""
    return {**BASELINE, "outlier_kt": BASELINE["outlier_ms"] * pp.MS_TO_KT}


# =============================================================================
# FILTRO DE KALMAN (T §3.3, S21 §III-A)
# =============================================================================
class AdaptedKalmanFilter:
    """M = I, Q = q·I, R = R0·I. Estado em kt, covariâncias em kt²."""

    def __init__(self, grid, q, r0, p0, x0, gate_kt):
        self.grid, self.q, self.r0, self.gate = grid, q, r0, gate_kt
        self.x = np.full(grid.n, float(x0))
        self.P = np.eye(grid.n) * p0
        # Rejeições do gate por nível (nível da observação pela Eq. 3.32)
        self.gated_by_level = np.zeros(grid.n, dtype=int)

    @property
    def n_gated(self):
        return int(self.gated_by_level.sum())

    def forecast(self, n_steps):
        """T Eqs. (3.24)-(3.25) aplicadas n_steps vezes: x inalterado, P += n·q·I.
        Exatamente equivalente a iterar passo a passo porque M = I e Q é constante."""
        if n_steps > 0:
            self.P[np.diag_indices_from(self.P)] += n_steps * self.q

    def analysis(self, heights_ft, values_kt):
        """Passo de análise, T §3.3: gate (3.20), ganho (3.21), estado (3.22), covariância (3.23)."""
        heights = np.asarray(heights_ft, dtype=float)
        H = np.array([self.grid.h_row(h) for h in heights])
        y = np.asarray(values_kt, dtype=float)

        # Eq. (3.20): rejeita |y_i - h_i x^f| > 10 m/s; removem-se as linhas de y e H
        # e a linha/coluna de R correspondentes.
        keep = np.abs(y - H @ self.x) <= self.gate
        for h in heights[~keep]:
            self.gated_by_level[self.grid.nearest_level(h)] += 1
        if not keep.any():
            return  # todas rejeitadas: passo de análise omitido (T §3.3)
        H, y = H[keep], y[keep]
        R = np.eye(len(y)) * self.r0

        # Eq. (3.21): K = P Hᵀ (H P Hᵀ + R)⁻¹.
        # Calculado resolvendo K·S = P·Hᵀ, sem inversa explícita (mesmo resultado, mais estável).
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T

        self.x = self.x + K @ (y - H @ self.x)                  # Eq. (3.22)
        self.P = (np.eye(self.grid.n) - K @ H) @ self.P         # Eq. (3.23)


def run_filter(obs_by_step, start_step, query_steps, grid, q, cfg, exclude=None):
    """Corre o filtro desde start_step.
    Devolve ({k: x^a_k} para cada k em query_steps, rejeições do gate por nível).
    exclude: identificador de passagem cujas observações são omitidas (leave-one-out)."""
    kf = AdaptedKalmanFilter(grid, q, cfg["r0_kt2"], cfg["p0_kt2"], cfg["x0_kt"], cfg["gate_kt"])
    query = set(query_steps)
    events = sorted(set(obs_by_step) | query)
    out, k_prev = {}, start_step                      # P^f no start_step é P0
    for k in events:
        if k < start_step:
            continue
        kf.forecast(k - k_prev)
        batch = [o for o in obs_by_step.get(k, []) if o.aircraft != exclude]
        if batch:
            kf.analysis([o.h_ft for o in batch], [o.headwind_kt for o in batch])
        if k in query:
            out[k] = kf.x.copy()
        k_prev = k
    return out, kf.gated_by_level.copy()


# =============================================================================
# MODELO BASELINE (T §3.4)
# =============================================================================
class EurocontrolBaseline:
    """Estado: estimativa por nível (kt), observações ativas por nível, validade."""

    def __init__(self, grid, x0, params):
        self.grid, self.p = grid, params
        self.x = np.full(grid.n, float(x0))           # x^b_{k,i}; x0 antes do 1.º valor válido
        self.valid = np.zeros(grid.n, dtype=bool)     # N_{k,i} ≥ 3 no passo atual
        self.active = [dict() for _ in range(grid.n)]  # id(obs) -> obs, por nível
        self.rejected = {}                             # id(obs) -> obs rejeitada (Eq. 3.33)

    def level_of(self, o):
        return self.grid.nearest_level(o.h_ft)        # Eq. (3.32)

    def add(self, o):
        self.active[self.level_of(o)][id(o)] = o

    def remove(self, o):
        self.active[self.level_of(o)].pop(id(o), None)

    def recompute(self, i):
        """Eqs. (3.33)–(3.34) no nível i com as observações atualmente na janela."""
        obs = list(self.active[i].values())
        if not obs:
            self.valid[i] = False
            return
        y = np.array([o.headwind_kt for o in obs])
        keep = np.abs(y - y.mean()) <= self.p["outlier_kt"]          # Eq. (3.33), 1 passagem
        self.rejected.update((id(o), o) for o, k in zip(obs, keep) if not k)
        if keep.sum() >= self.p["min_obs"]:
            self.x[i] = y[keep].mean()                                # Eq. (3.34), N ≥ 3
            self.valid[i] = True
        else:
            self.valid[i] = False                                     # mantém x^b_{k-1,i}


def exit_step(o, window_s):
    """Primeiro passo inteiro k em que t_k − window ≥ t_m, i.e. a observação sai (Eq. 3.31)."""
    return int(math.ceil(o.t_raw + window_s))


def run_baseline(obs_by_step, start_step, query_steps, grid, x0, params, exclude=None):
    """Corre o baseline desde start_step. Devolve ({k: x^b_k}, {k: N≥3 por nível},
    observações rejeitadas pela Eq. 3.33).
    Entrada de uma observação no passo o.step = ceil(t_m) (replay causal, T §3.1.3),
    saída no passo ceil(t_m + 600). exclude: passagem omitida (leave-one-out)."""
    enters, exits = {}, {}
    for k, batch in obs_by_step.items():
        for o in batch:
            if o.aircraft == exclude:
                continue
            enters.setdefault(k, []).append(o)
            exits.setdefault(exit_step(o, params["window_s"]), []).append(o)

    model = EurocontrolBaseline(grid, x0, params)
    query = set(query_steps)
    last_query = max(query)
    out, valid_at = {}, {}
    for k in sorted(set(enters) | set(exits) | query):
        if k < start_step:
            continue
        if k > last_query:                              # nada a reportar depois disto
            break
        touched = set()
        for o in exits.get(k, []):                     # saem antes de entrar (janela em k)
            model.remove(o)
            touched.add(model.level_of(o))
        for o in enters.get(k, []):
            model.add(o)
            touched.add(model.level_of(o))
        for i in touched:
            model.recompute(i)
        if k in query:
            out[k] = model.x.copy()
            valid_at[k] = model.valid.copy()
    return out, valid_at, list(model.rejected.values())


# =============================================================================
# BENCHMARK METAR (T §3.5)
# =============================================================================
def metar_benchmark_estimates(metar_reports, query_steps, n_levels, rwy_bearing_deg):
    """x^M_{k,i}: headwind de superfície (Eq. 3.1) do METAR mais recente emitido até k,
    igual em todos os níveis. Não usa observações de aeronaves."""
    out = {}
    for k in query_steps:
        hw = pp.surface_headwind_kt(pp.metar_valid_at(metar_reports, k), rwy_bearing_deg)
        out[k] = np.full(n_levels, hw)
    return out


# =============================================================================
# INTERFACE COMUM
# =============================================================================
@dataclass(frozen=True)
class Model:
    key: str                  # identificador curto (CSV)
    label: str                # legenda das figuras (inglês britânico)
    estimate: Callable        # (obs_by_step, start_step, query_steps, exclude=None) -> {k: x_k}


def kalman_model(grid, q, run_cfg):
    """run_cfg tem de conter r0_kt2, p0_kt2, gate_kt e x0_kt."""
    def estimate(obs_by_step, start_step, query_steps, exclude=None):
        return run_filter(obs_by_step, start_step, query_steps, grid, q, run_cfg, exclude)[0]
    return Model("kalman", f"Kalman filter (q = {q:g} kt²/s)", estimate)


def baseline_model(grid, x0_kt, params=None):
    params = params or baseline_params()

    def estimate(obs_by_step, start_step, query_steps, exclude=None):
        return run_baseline(obs_by_step, start_step, query_steps, grid, x0_kt, params, exclude)[0]
    return Model("baseline", "Baseline (EUROCONTROL)", estimate)


def metar_model(grid, metar_reports, rwy_bearing_deg):
    def estimate(obs_by_step, start_step, query_steps, exclude=None):
        return metar_benchmark_estimates(metar_reports, query_steps, grid.n, rwy_bearing_deg)
    return Model("metar", "METAR benchmark", estimate)


def evaluation_models(grid, x0_kt, metar_reports, rwy_bearing_deg, kalman_params=KALMAN):
    """Os três modelos do Cap. 5, pela ordem das tabelas, com o mesmo x0 (IPMA) para o
    filtro e o baseline."""
    run_cfg = {**kalman_derived(kalman_params), "x0_kt": x0_kt}
    return [baseline_model(grid, x0_kt),
            kalman_model(grid, kalman_params["q_kt2_s"], run_cfg),
            metar_model(grid, metar_reports, rwy_bearing_deg)]
