"""
Avaliação dos modelos (T §3.5): referência das aeronaves, avaliação nowcast e operacional
e métricas. O MESMO código serve os três modelos (filtro, baseline e benchmark METAR) e
a afinação de q: o modelo entra como argumento (modelos.Model).

Referências:
  [T] Tese WindOnFinal: §3.5, Eqs. (3.35)–(3.38).

Origem: reference_by_level e as regras de nowcast/operacional vêm de
'kalman_tuning_1julho.py' e 'baseline_1julho.py' (22-10-2026), unificadas sem alterar a
lógica. Formato dos erros: {passagem: {nível: e}} no nowcast e {passagem: {"mean": e}}
no operacional. Sinal: e = estimativa − referência (positivo = sobrestima o headwind).
"""

import math

import numpy as np

import preprocessamento as pp


def reference_by_level(aircraft_obs, grid):
    """T Eq. (3.35): média das observações da passagem em cada nível (nível pela Eq. 3.32)."""
    by_level = pp.group_by(aircraft_obs, lambda o: grid.nearest_level(o.h_ft))
    return {i: np.mean([o.headwind_kt for o in v]) for i, v in by_level.items()}, by_level


def nowcast_errors(model, obs, obs_by_step, start_step, test_ids, grid):
    """T Eq. (3.36), leave-one-passage-out: o modelo corre sem as observações da passagem j
    e é avaliado nos passos em que j as transmitiu. Devolve {j: {i: e_{j,i}}}."""
    by_ac = pp.group_by(obs, lambda o: o.aircraft)
    errors = {}
    for j in test_ids:
        ref, by_level = reference_by_level(by_ac[j], grid)
        est = model.estimate(obs_by_step, start_step, [o.step for o in by_ac[j]], exclude=j)
        errors[j] = {i: np.mean([est[o.step][i] for o in by_level[i]]) - ref[i] for i in ref}
    return errors


def operational_errors(model, obs, obs_by_step, start_step, test_ids, grid):
    """T Eq. (3.37): estimativa em k_j = (1.º passo de j) − 1, sem dados de j por
    construção, média sobre os níveis B_j. Devolve {j: {"mean": e_j}}."""
    by_ac = pp.group_by(obs, lambda o: o.aircraft)
    k_j = {j: min(o.step for o in by_ac[j]) - 1 for j in test_ids}
    est = model.estimate(obs_by_step, start_step, list(k_j.values()))
    errors = {}
    for j in test_ids:
        ref, _ = reference_by_level(by_ac[j], grid)
        lv = sorted(ref)
        errors[j] = {"mean": np.mean(est[k_j[j]][lv]) - np.mean([ref[i] for i in lv])}
    return errors


def metrics(errors):
    """T Eq. (3.38): bias, RMSE e MAE sobre todos os erros (todos os pares (j, i) no
    nowcast; uma por passagem no operacional)."""
    e = np.array([v for d in errors.values() for v in d.values()], dtype=float)
    return {"bias": float(e.mean()), "rmse": math.sqrt(float(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e))), "n": len(e)}


def per_level_metrics(errors, grid):
    """Métricas nowcast por nível; n = nº de passagens de teste com observação no nível."""
    rows = []
    for i in range(grid.n):
        e = [d[i] for d in errors.values() if i in d]
        m = metrics({0: dict(enumerate(e))}) if e else {"bias": np.nan, "rmse": np.nan,
                                                         "mae": np.nan, "n": 0}
        rows.append({"level_ft": int(grid.levels_ft[i]), **m})
    return rows


# =============================================================================
# TESTES DE SANIDADE
# =============================================================================
def run_selftests():
    from cenarios import GRID, AltitudeGrid
    from modelos import Model

    grid = AltitudeGrid.from_config(GRID)
    truth = lambda h: 5.0 + 0.004 * h

    # Duas passagens sintéticas sem ruído, uma mensagem a cada 5 s.
    obs = []
    for j, t0 in (("A#1", 0), ("B#1", 1000)):
        for n, h in enumerate(np.arange(3990.0, 410.0, -60.0)):
            obs.append(pp.Observation(t0 + 5 * n, j, j[0], t0 + 5 * n, h,
                                      truth(grid.levels_ft[grid.nearest_level(h)])))
    obs_by_step = pp.group_by(obs, lambda o: o.step)
    ids = ["A#1", "B#1"]

    # 1. Estimador perfeito -> erro zero; desvio constante -> bias = RMSE = MAE = desvio.
    perfect = Model("p", "p", lambda ob, s, ks, exclude=None:
                    {k: truth(grid.levels_ft) for k in ks})
    offset = Model("o", "o", lambda ob, s, ks, exclude=None:
                   {k: truth(grid.levels_ft) + 2.0 for k in ks})
    m = metrics(nowcast_errors(perfect, obs, obs_by_step, 0, ids, grid))
    assert m["rmse"] < 1e-12 and m["n"] == 2 * 36
    m = metrics(nowcast_errors(offset, obs, obs_by_step, 0, ids, grid))
    assert abs(m["bias"] - 2.0) < 1e-12 and abs(m["rmse"] - 2.0) < 1e-12 and abs(m["mae"] - 2.0) < 1e-12
    m = metrics(operational_errors(offset, obs, obs_by_step, 0, ids, grid))
    assert abs(m["bias"] - 2.0) < 1e-12 and m["n"] == 2

    # 2. Leave-one-out: o modelo recebe exclude = j no nowcast, e nada no operacional.
    seen = []
    spy = Model("s", "s", lambda ob, s, ks, exclude=None:
                seen.append(exclude) or {k: np.zeros(grid.n) for k in ks})
    nowcast_errors(spy, obs, obs_by_step, 0, ids, grid)
    operational_errors(spy, obs, obs_by_step, 0, ids, grid)
    assert seen == ids + [None]

    # 3. Operacional: estimativa pedida no passo anterior à 1.ª observação de j.
    asked = []
    spy = Model("s", "s", lambda ob, s, ks, exclude=None:
                asked.extend(ks) or {k: np.zeros(grid.n) for k in ks})
    operational_errors(spy, obs, obs_by_step, 0, ids, grid)
    assert sorted(asked) == [-1, 999]

    # 4. Por nível: cada nível tem uma observação de cada passagem.
    rows = per_level_metrics(nowcast_errors(offset, obs, obs_by_step, 0, ids, grid), grid)
    assert all(r["n"] == 2 and abs(r["rmse"] - 2.0) < 1e-12 for r in rows)
    print("Testes de sanidade de avaliacao.py: todos passaram.")


if __name__ == "__main__":
    run_selftests()
