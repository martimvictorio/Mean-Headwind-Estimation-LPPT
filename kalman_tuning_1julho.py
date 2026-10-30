"""
Afinação do parâmetro q do Filtro de Kalman (AKF) — cenário de 1 de julho de 2026.

Referências:
  [T]   Tese WindOnFinal, Cap. 3 (metodologia) e §4.3.2 (implementação do filtro).
  [S21] Sun, Marinescu, Olivares & Staffetti (2021), ATM Seminar, Sec. III-A e IV.

O que este script faz:
  1. Pré-processamento das mensagens CAT062 (T §3.2.1) e cálculo do headwind (T §3.2.2),
     através do módulo partilhado 'preprocessamento.py'.
  2. Filtro de Kalman com M = I e Q = q·I (T §3.3 / S21 §III-A), passo Δt = 1 s.
  3. Avaliação nowcast (leave-one-aircraft-out) e operacional (T §3.5).
  4. Varrimento de q e tabela de RMSE / MAE / bias.
  5. No fim, rejeições do gate (Eq. 3.20) por nível e no total, na corrida completa.

Alterações de 21-10-2026:
  - O pré-processamento (tempo, METAR, física, critérios, passagens) saiu deste ficheiro
    para 'preprocessamento.py', sem alterações de lógica. Os nomes continuam acessíveis
    como kt.<nome> para 'baseline_1julho.py' e 'kalman_visualizacao_q_1julho.py'.
  - sweep_q passou a devolver (results, gated_by_q); antes não tinha return e main()
    falhava ao desempacotar.
  - A grelha de altitude (AltitudeGrid e parâmetros) passou para 'cenarios.py', sem
    alterações, porque também é usada na avaliação (Eqs. 3.32, 3.35) e nas figuras.
  - O filtro (AdaptedKalmanFilter, run_filter) e os seus parâmetros passaram para
    'modelos.py', e a avaliação (referência, nowcast, operacional, métricas) para
    'avaliacao.py', sem alterações de lógica: a afinação usa assim exatamente o mesmo
    código de avaliação que os resultados do Cap. 5.

Alteração de 29-10-2026: 'tod' passou a ser lido como UTC (preprocessamento.py). O início
da simulação é 12:00 UTC (antes 11:00 UTC) e x0 vem do METAR das 12:00Z (antes 11:00Z).
Removidos da reexportação lisbon_offset_h, local_tod_to_utc_s, lisbon_offset_h_at e
utc_s_to_local_s, que deixaram de existir.

Vento à superfície:
  - Entra apenas no estado inicial x0 (T §4.3.2).
  - Neste cenário vem do METAR porque não há dados IPMA para 1 de julho.
  - Nos cenários de avaliação, x0 deve vir do IPMA (T §3.1.2): ver "x0_kt" no CONFIG.

Uso:
  python kalman_tuning_1julho.py --csv dados_1julho_12h16h.csv --metar METAR_julho.txt
  python kalman_tuning_1julho.py --selftest
"""

import argparse
import csv
import os

import numpy as np

import avaliacao
import preprocessamento
from avaliacao import reference_by_level  # noqa: F401  (kt.<nome>)
from cenarios import EVALUATION, GRID, AltitudeGrid, save_config  # noqa: F401  (kt.<nome>)
from modelos import KALMAN, AdaptedKalmanFilter, kalman_derived, kalman_model, run_filter  # noqa: F401
# Reexportados para os scripts que os usam como kt.<nome> (baseline, visualização).
from preprocessamento import (  # noqa: F401
    PREPROCESSING, P0_HPA, T0_K, LAMBDA, R_AIR, G0, FT_TO_M, NM_TO_M, MS_TO_KT,
    fmt_hhmm,
    MetarReport, load_metar_schedule, metar_valid_at, surface_headwind_kt,
    static_pressure_hpa, true_altitude_m, true_airspeed_kt, bearing_deg,
    Observation, parse_tod_s, observation_from_row, load_observations, assign_passages,
    group_by,
)

# =============================================================================
# CONFIGURAÇÃO CENTRALIZADA
# =============================================================================
# Pré-processamento: preprocessamento.PREPROCESSING (geometria, limites de altitude,
# separação de passagens). Aqui ficam só a data, a grelha, o filtro e a avaliação.
# Os valores derivados (R0, P0, gate em kt, x0) são calculados num único sítio:
# derived_params() e derive_run_config().
CONFIG = {
    **PREPROCESSING,

    # --- Dados ---
    # Data do cenário (AAAAMMDD). Serve para escolher as linhas METAR do dia ('tod' e
    # METAR estão ambos em UTC).
    "scenario_date": "20260701",

    # --- Grelha de altitude: definida em cenarios.GRID ---
    **GRID,

    # --- Filtro de Kalman: σ, P0 e gate definidos em modelos.KALMAN (q varia aqui) ---
    **{k: v for k, v in KALMAN.items() if k != "q_kt2_s"},
    # x0 = headwind à superfície em todos os níveis (T §4.3.2).
    #   None  -> calculado a partir do METAR válido no instante inicial
    #            (usado em 1 de julho, por falta de dados IPMA).
    #   valor -> headwind IPMA em kt (média dos sensores 3A/3B, T §3.1.2), a usar
    #            nos cenários de avaliação.
    "x0_kt": None,

    # --- Avaliação (T §3.5): warm-up definido em cenarios.EVALUATION ---
    **EVALUATION,

    # --- Varrimento de q [kt²/s] ---
    # 0.126 ≈ valor de S21 (Q = I (m/s)² por passo de 30 s) convertido para kt² por 1 s.
    "q_grid": [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.126, 0.2, 0.5, 1.0],
    # Critério de seleção: DECISÃO EM ABERTO ("nowcast_rmse" ou "operational_rmse").
    "selection_metric": "nowcast_rmse",

    # --- Reprodutibilidade ---
    "seed": 20260701,
}


def derived_params(base_cfg):
    """Parâmetros do filtro derivados do CONFIG. Único sítio onde são calculados."""
    return {**base_cfg, **kalman_derived(base_cfg)}


def derive_run_config(base_cfg, metar_reports, start_step):
    """Configuração completa de uma corrida: parâmetros derivados + x0. Não altera base_cfg."""
    cfg = derived_params(base_cfg)
    if cfg["x0_kt"] is None:
        report = metar_valid_at(metar_reports, start_step)
        cfg["x0_kt"] = surface_headwind_kt(report, cfg["rwy_true_bearing_deg"])
        cfg["x0_source"] = f"METAR {report.t_utc_s // 3600:02d}{report.t_utc_s % 3600 // 60:02d}Z (sem dados IPMA)"
    else:
        cfg["x0_source"] = "valor fixo no CONFIG (IPMA)"
    return cfg


# =============================================================================
# AVALIAÇÃO (T §3.5)
# =============================================================================
def nowcast_errors(obs, obs_by_step, start_step, test_ids, grid, q, cfg):
    """T Eq. (3.36), leave-one-aircraft-out, com o código comum de avaliacao.py."""
    return avaliacao.nowcast_errors(kalman_model(grid, q, cfg), obs, obs_by_step,
                                    start_step, test_ids, grid)


def operational_errors(obs, obs_by_step, start_step, test_ids, grid, q, cfg):
    """T Eq. (3.37), com o código comum de avaliacao.py. Devolve também as rejeições do
    gate por nível na corrida completa do cenário (todas as aeronaves)."""
    errors = avaliacao.operational_errors(kalman_model(grid, q, cfg), obs, obs_by_step,
                                          start_step, test_ids, grid)
    _, gated_by_level = run_filter(obs_by_step, start_step, [], grid, q, cfg)
    return errors, gated_by_level


metrics = avaliacao.metrics   # T Eq. (3.38)


# =============================================================================
# REJEIÇÕES DO GATE (Eq. 3.20)
# =============================================================================
def report_gate_rejections(obs, grid, gated_by_q, csv_path):
    """Imprime e guarda as rejeições do gate por nível e no total, para cada q.
    Contagens da corrida completa do cenário (todas as aeronaves, sem leave-one-out).
    'n_obs' = observações aceites no pré-processamento em cada nível (Eq. 3.32)."""
    n_obs = np.zeros(grid.n, dtype=int)
    for o in obs:
        n_obs[grid.nearest_level(o.h_ft)] += 1
    qs = list(gated_by_q)
    header = ["level_ft", "n_obs"] + [f"q={q:g}" for q in qs]
    rows = [[f"{grid.levels_ft[i]:.0f}", int(n_obs[i])] + [int(gated_by_q[q][i]) for q in qs]
            for i in range(grid.n)]
    rows.append(["TOTAL", int(n_obs.sum())] + [int(gated_by_q[q].sum()) for q in qs])

    widths = [max(len(str(r[c])) for r in [header] + rows) for c in range(len(header))]
    print("\nRejeições do gate (Eq. 3.20) por nível, corrida completa do cenário:")
    for r in [header] + rows:
        print("  ".join(str(v).rjust(w) for v, w in zip(r, widths)))

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Rejeições guardadas em: {csv_path}")


# =============================================================================
# VARRIMENTO DE q
# =============================================================================
def sweep_q(obs, start_step, cfg, grid):
    obs_by_step = group_by(obs, lambda o: o.step)
    first_step = {j: min(o.step for o in v) for j, v in group_by(obs, lambda o: o.aircraft).items()}
    test_ids = sorted(j for j, k in first_step.items() if k >= start_step + cfg["warmup_s"])
    print(f"Passagens no total: {len(first_step)} | aeronaves de teste (pós warm-up): {len(test_ids)}")

    results, gated_by_q = [], {}
    for q in cfg["q_grid"]:
        now = nowcast_errors(obs, obs_by_step, start_step, test_ids, grid, q, cfg)
        ops, gated_by_q[q] = operational_errors(obs, obs_by_step, start_step, test_ids, grid, q, cfg)
        n_gated = int(gated_by_q[q].sum())
        m_now, m_ops = metrics(now), metrics(ops)
        results.append({"q": q, "n_gated_full_run": n_gated,
                        **{f"nowcast_{k}": v for k, v in m_now.items()},
                        **{f"operational_{k}": v for k, v in m_ops.items()}})
        r = results[-1]
        print(f"q={q:<7g} | NOWCAST RMSE {r['nowcast_rmse']:5.2f} MAE {r['nowcast_mae']:5.2f} "
              f"bias {r['nowcast_bias']:+5.2f} (n={r['nowcast_n']}) | OPER RMSE {r['operational_rmse']:5.2f} "
              f"MAE {r['operational_mae']:5.2f} bias {r['operational_bias']:+5.2f} "
              f"(n={r['operational_n']}) | gated {n_gated}")
    return results, gated_by_q


def save_results(results, cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path_csv = os.path.join(out_dir, "q_sweep_1julho.csv")
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    save_config(cfg, out_dir)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[AVISO] matplotlib indisponível: gráfico não gerado.")
        return path_csv
    import estilo_figuras as ef
    q = [r["q"] for r in results]
    best_q = min(results, key=lambda r: r[cfg["selection_metric"]])["q"]
    color = ef.INK                                   # RMSE, MAE e q selecionado a preto
    sel_kind, sel_metric = cfg["selection_metric"].split("_")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, kind, name in zip(axes, ("nowcast", "operational"), ("Nowcast", "Operational")):
        ax.axvline(best_q, color=color, linewidth=1.0, linestyle=":", zorder=2,
                   label=f"Selected q = {best_q:g} kt²/s (minimum {sel_kind} {sel_metric.upper()})")
        ax.plot(q, [r[f"{kind}_rmse"] for r in results], color=color, linestyle="-",
                marker="o", markersize=4, linewidth=ef.MODEL_LINEWIDTH, zorder=3, label="RMSE")
        ax.plot(q, [r[f"{kind}_mae"] for r in results], color=color, linestyle="--",
                marker="s", markersize=4, linewidth=ef.MODEL_LINEWIDTH, zorder=3, label="MAE")
        ax.set_xscale("log")
        ax.set_xlabel("q [kt²/s]")
        ax.set_ylim(bottom=0)
        ef.panel_label(ax, name)
        ef.style_axes(ax)
    axes[0].set_ylabel("Error [kt]")
    fig.tight_layout(rect=ef.bottom_legend(fig, axes[0], ncol=3, bottom=0.09))
    fig.savefig(os.path.join(out_dir, "q_sweep_1julho.png"), dpi=150)
    return path_csv


# =============================================================================
# TESTES DE SANIDADE
# =============================================================================
def run_selftests(base_cfg):
    cfg = {**derived_params(base_cfg), "x0_kt": 0.0}     # testes independentes do METAR
    grid = AltitudeGrid.from_config(cfg)
    assert grid.n == 36 and grid.levels_ft[0] == 450 and grid.levels_ft[-1] == 3950

    # 1–2 e 8 (altitude, TAS, leitura de 'tod' em UTC) estão em preprocessamento.run_selftests().
    preprocessamento.run_selftests()

    # 3. Linhas de H somam 1 e reproduzem o exemplo da tese (1790 ft -> 0.6/0.4 em 1750/1850)
    for h in (300.0, 450.0, 1790.0, 3950.0, 4000.0):
        assert abs(grid.h_row(h).sum() - 1.0) < 1e-12
    row = grid.h_row(1790.0)
    i = int(np.where(grid.levels_ft == 1750)[0][0])
    assert abs(row[i] - 0.6) < 1e-12 and abs(row[i + 1] - 0.4) < 1e-12

    # 4. Nível mais próximo (Eq. 3.32)
    assert grid.nearest_level(400.1) == 0 and grid.nearest_level(500.0) == 0
    assert grid.nearest_level(500.1) == 1 and grid.nearest_level(3999.9) == grid.n - 1

    # 5. Caso sintético com solução conhecida: perfil linear constante no tempo + ruído
    rng = np.random.default_rng(cfg["seed"])
    truth = lambda h: 5.0 + 0.004 * h                     # kt
    obs, step = [], 0
    for ac in range(40):                                  # 40 aeronaves, 1 a cada 180 s
        for h in np.arange(3990.0, 410.0, -60.0):         # 1 mensagem / 5 s
            obs.append(Observation(step, f"A{ac}", f"A{ac}", step, h,
                                   truth(h) + rng.normal(0, cfg["sigma_obs_kt"])))
            step += 5
        step += 180
    est, _ = run_filter(group_by(obs, lambda o: o.step), 0, [step], grid, 0.001, cfg)
    err = est[step] - truth(grid.levels_ft)
    assert np.sqrt(np.mean(err ** 2)) < 1.5, f"RMSE sintético alto: {np.sqrt(np.mean(err**2)):.2f}"

    # 6. Diagnóstico: com a Eq. (3.23), P continua simétrica (até arredondamento) e
    #    definida positiva após muitas atualizações. Não há correção, só verificação.
    kf = AdaptedKalmanFilter(grid, 0.05, cfg["r0_kt2"], cfg["p0_kt2"], 0.0, 1e9)
    for o in obs[:2000]:
        kf.forecast(5)
        kf.analysis([o.h_ft], [o.headwind_kt])
    assert np.abs(kf.P - kf.P.T).max() < 1e-8
    assert np.linalg.eigvalsh(0.5 * (kf.P + kf.P.T)).min() > 0

    # 7. Gate: observação a 30 kt da previsão é rejeitada e contada no nível certo (1950 ft)
    kf = AdaptedKalmanFilter(grid, 0.05, cfg["r0_kt2"], cfg["p0_kt2"], 0.0, cfg["gate_kt"])
    kf.analysis([2000.0], [30.0])
    i_1950 = int(np.where(grid.levels_ft == 1950)[0][0])
    assert kf.n_gated == 1 and kf.gated_by_level[i_1950] == 1 and np.all(kf.x == 0.0)

    print("Todos os testes de sanidade passaram.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="dados_1julho_12h16h.csv")
    ap.add_argument("--metar", default="METAR_julho.txt")
    ap.add_argument("--out", default="resultados_q_1julho")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    run_selftests(CONFIG)
    if args.selftest:
        return

    grid = AltitudeGrid.from_config(CONFIG)
    metar = load_metar_schedule(args.metar, CONFIG["scenario_date"])
    obs, n_rows, start_step = load_observations(args.csv, metar, CONFIG)
    obs = assign_passages(obs, CONFIG["passage_gap_s"])
    cfg = derive_run_config(CONFIG, metar, start_step)

    print(f"x0 = {cfg['x0_kt']:.2f} kt ({cfg['x0_source']}) | "
          f"P0 = {cfg['p0_c']:g}·σ² = {cfg['p0_kt2']:.1f} kt² | gate = {cfg['gate_kt']:.2f} kt")
    print(f"Mensagens lidas: {n_rows} | observações aceites: {len(obs)} | "
          f"início da simulação: {fmt_hhmm(start_step)} UTC")

    results, gated_by_q = sweep_q(obs, start_step, cfg, grid)
    key = cfg["selection_metric"]
    best = min(results, key=lambda r: r[key])
    print(f"\nq com menor {key}: {best['q']} kt²/s ({key} = {best[key]:.3f} kt)")
    print(f"Resultados guardados em: {save_results(results, cfg, args.out)}")
    report_gate_rejections(obs, grid, gated_by_q, os.path.join(args.out, "rejeicoes_gate_1julho.csv"))


if __name__ == "__main__":
    main()
