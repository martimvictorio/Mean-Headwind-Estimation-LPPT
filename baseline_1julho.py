"""
Implementação do baseline EUROCONTROL (média móvel por nível) — cenário de 1 de julho de 2026.

Objetivo: verificar se o baseline está bem implementado. Não inclui comparação com outros modelos.

Referências:
  [T]    Tese WindOnFinal, §3.4 (baseline, Eqs. 3.31–3.34), §3.5 (avaliação, Eqs. 3.35–3.38).
  [EC7]  EUROCONTROL Guidelines on TBS with ORD (2024), via T §2.2.2.

O que este script faz:
  1. Reutiliza SEM ALTERAÇÕES o pré-processamento, a conversão de tempo, a grelha, o x0
     (METAR válido no início) e a definição das passagens de 'kalman_tuning_1julho.py'.
  2. Baseline (T Eq. 3.34): em cada passo k e nível i, média das observações
       - recebidas nos últimos 10 min: t_k − 600 s < t_m ≤ t_k          (Eq. 3.31)
       - do nível mais próximo:        h_i − 50 < h_m ≤ h_i + 50 ft      (Eq. 3.32)
       - a ≤ 10 m/s da média de todas as que cumprem 3.31 e 3.32       (Eq. 3.33)
     se houver N ≥ 3; caso contrário mantém o último valor válido (x0 antes do primeiro).
  3. Avaliação nowcast (leave-one-aircraft-out) e operacional (T §3.5): RMSE e MAE.
  4. Rejeições da Eq. 3.33 por nível, na corrida completa.
  5. Figuras (legendas em inglês britânico) e CSV/JSON de resultados.

Alteração de 22-10-2026: o modelo (EurocontrolBaseline, exit_step, run_baseline e
parâmetros) passou para 'modelos.py' e a avaliação para 'avaliacao.py', sem alterações de
lógica, para que o Cap. 5 use exatamente o mesmo código.

Alteração de 29-10-2026: instantes e eixos em UTC ('tod' é UTC); removida a conversão
para hora local (estilo_figuras.to_local_clock deixou de existir).

Implementação orientada a eventos: o conjunto de observações na janela de um nível só muda
quando uma observação entra (passo ceil(t_m)) ou sai (passo ceil(t_m + 600)). Entre eventos
a estimativa é constante, pelo que o resultado é exatamente igual a recalcular a cada 1 s.
O teste de sanidade 6 verifica esta equivalência contra uma implementação passo a passo.

Uso:
  python baseline_1julho.py --csv dados_1julho_12h16h.csv --metar METAR_julho.txt
  python baseline_1julho.py --selftest
Requer 'kalman_tuning_1julho.py' na mesma pasta (pré-processamento partilhado).
"""

import argparse
import csv
import math
import os

import numpy as np

import avaliacao
import estilo_figuras as ef
import kalman_tuning_1julho as kt
from modelos import (BASELINE, EurocontrolBaseline, baseline_model,  # noqa: F401
                     baseline_params, exit_step, run_baseline)

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================
# Parâmetros do baseline: modelos.BASELINE (único sítio onde são definidos).
VIS = {
    "snapshot_hour": 3.0,     # instante do perfil (h desde o início da simulação)
    "mean_sample_s": 10,      # resolução temporal da série do vento médio
}


# =============================================================================
# AVALIAÇÃO (T §3.5)
# =============================================================================
def nowcast_errors(obs, obs_by_step, start_step, test_ids, grid, x0, params):
    """T Eq. (3.36), com o código comum de avaliacao.py. Devolve {aeronave: {nível: e}}."""
    return avaliacao.nowcast_errors(baseline_model(grid, x0, params), obs, obs_by_step,
                                    start_step, test_ids, grid)


def operational_errors(obs, obs_by_step, start_step, test_ids, grid, x0, params):
    """T Eq. (3.37), com o código comum de avaliacao.py."""
    return avaliacao.operational_errors(baseline_model(grid, x0, params), obs, obs_by_step,
                                        start_step, test_ids, grid)


def rmse_mae(errors):
    """T Eq. (3.38), só RMSE e MAE."""
    e = np.array([v for d in errors.values() for v in d.values()], dtype=float)
    return {"rmse": math.sqrt(np.mean(e ** 2)), "mae": np.mean(np.abs(e)), "n": len(e)}


def per_level_rmse(errors, grid):
    """RMSE nowcast por nível (n = nº de aeronaves de teste com observação nesse nível)."""
    rows = []
    for i in range(grid.n):
        e = np.array([v[i] for v in errors.values() if i in v])
        rows.append((grid.levels_ft[i], math.sqrt(np.mean(e ** 2)) if len(e) else np.nan, len(e)))
    return rows


# =============================================================================
# REJEIÇÕES DA EQ. 3.33 POR NÍVEL
# =============================================================================
def rejections_by_level(obs, rejected, grid):
    """Por nível: observações aceites no pré-processamento, observações rejeitadas pela
    Eq. 3.33 e nº de aeronaves (passagens) com pelo menos uma observação rejeitada."""
    n_obs = np.zeros(grid.n, dtype=int)
    for o in obs:
        n_obs[grid.nearest_level(o.h_ft)] += 1
    n_rej = np.zeros(grid.n, dtype=int)
    ac_rej = [set() for _ in range(grid.n)]
    for o in rejected:
        i = grid.nearest_level(o.h_ft)
        n_rej[i] += 1
        ac_rej[i].add(o.aircraft)
    rows = [[f"{grid.levels_ft[i]:.0f}", int(n_obs[i]), int(n_rej[i]), len(ac_rej[i])]
            for i in range(grid.n)]
    rows.append(["TOTAL", int(n_obs.sum()), int(n_rej.sum()),
                 len({o.aircraft for o in rejected})])
    return ["level_ft", "n_obs", "rejected_obs", "rejected_aircraft"], rows


def print_table(title, header, rows):
    widths = [max(len(str(r[c])) for r in [header] + rows) for c in range(len(header))]
    print(f"\n{title}")
    for r in [header] + rows:
        print("  ".join(str(v).rjust(w) for v, w in zip(r, widths)))


# =============================================================================
# FIGURAS (inglês britânico)
# =============================================================================
def observed_mean_per_aircraft(obs, grid):
    """Headwind médio observado por aeronave (média dos níveis, Eqs. 3.35 e 3.37),
    no instante da sua primeira observação aceite."""
    t, v = [], []
    for ac_obs in kt.group_by(obs, lambda o: o.aircraft).values():
        ref, _ = kt.reference_by_level(ac_obs, grid)
        t.append(min(o.step for o in ac_obs))
        v.append(np.mean(list(ref.values())))
    return np.array(t), np.array(v)


def make_figures(cfg, grid, obs, start_step, end_step, params, nowcast, out_dir):
    """Figuras com o estilo comum (estilo_figuras.py): sem títulos, identificação dentro do
    painel, legenda por baixo, cores fixas por modelo."""
    import matplotlib.pyplot as plt

    style = ef.MODEL_STYLE["baseline"]
    obs_by_step = kt.group_by(obs, lambda o: o.step)
    snap = start_step + int(VIS["snapshot_hour"] * 3600)
    if snap > end_step:
        raise ValueError("O instante do perfil está depois do fim dos dados.")
    t = list(range(start_step, end_step + 1, VIS["mean_sample_s"]))
    est, _, _ = run_baseline(obs_by_step, start_step, t + [snap], grid, cfg["x0_kt"], params)
    label = "Baseline (EUROCONTROL)"

    # 1. Perfil no instante T, com as observações que o baseline tem na janela
    fig, ax = plt.subplots(figsize=(7.5, 8))
    win = [o for o in obs if snap - params["window_s"] < o.t_raw <= snap]
    ax.scatter([o.headwind_kt for o in win], [o.h_ft for o in win], zorder=2,
               label="Observations in the last 10 min", **ef.OBS_POINTS)
    ax.plot(est[snap], grid.levels_ft, linewidth=ef.MODEL_LINEWIDTH, zorder=3, label=label, **style)
    ef.panel_label(ax, f"{kt.fmt_hhmm(snap)} UTC (T = {VIS['snapshot_hour']:g} h)")
    ax.set_xlabel("Headwind [kt]")
    ax.set_ylabel("True altitude [ft]")
    ax.set_ylim(*ef.ALTITUDE_LIM_FT)
    ef.style_axes(ax)
    fig.tight_layout(rect=ef.bottom_legend(fig, ax, ncol=2, bottom=0.06))
    fig.savefig(os.path.join(out_dir, "perfil_baseline_T3h.png"), dpi=150)

    # 2. Vento médio ao longo do tempo (Eq. 3.30 sem o termo de superfície)
    t_ac, v_ac = observed_mean_per_aircraft(obs, grid)
    fig, ax = plt.subplots(figsize=(11, 5))
    t0 = start_step
    ax.axvspan(t0, t0 + cfg["warmup_s"], label=f"Warm-up period ({cfg['warmup_s'] // 60:g} min)",
               **ef.WARMUP_SPAN)
    ax.plot(np.array(t), [est[k].mean() for k in t],
            linewidth=ef.MODEL_LINEWIDTH, zorder=3, label=label, **style)
    ax.scatter(t_ac, v_ac, zorder=4,
               label="Mean headwind observed by each passage", **ef.OBS_MEAN_POINTS)
    ef.format_time_axis(ax)
    ax.set_ylabel("Mean headwind of the 36 levels [kt]")
    ef.style_axes(ax)
    fig.tight_layout(rect=ef.bottom_legend(fig, ax, ncol=3, bottom=0.08))
    fig.savefig(os.path.join(out_dir, "headwind_medio_baseline.png"), dpi=150)

    # 3. RMSE nowcast por nível
    rows = per_level_rmse(nowcast, grid)
    fig, ax = plt.subplots(figsize=(6.5, 8))
    ax.plot([r[1] for r in rows], [r[0] for r in rows], marker="o", markersize=3,
            linewidth=ef.MODEL_LINEWIDTH, label=label, **style)
    ax.set_xlabel("Nowcast RMSE [kt]")
    ax.set_ylabel("Altitude level [ft]")
    ax.set_xlim(left=0)
    ax.set_ylim(*ef.ALTITUDE_LIM_FT)
    ef.style_axes(ax)
    fig.tight_layout(rect=ef.bottom_legend(fig, ax, ncol=1, bottom=0.05))
    fig.savefig(os.path.join(out_dir, "rmse_por_nivel_baseline.png"), dpi=150)
    return plt


# =============================================================================
# GRAVAÇÃO
# =============================================================================
def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def save_outputs(metrics, rej_header, rej_rows, nowcast, operational, grid, cfg, params, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    write_csv(os.path.join(out_dir, "metricas_baseline.csv"), ["evaluation", "rmse", "mae", "n"],
              [[k, m["rmse"], m["mae"], m["n"]] for k, m in metrics.items()])
    write_csv(os.path.join(out_dir, "rejeicoes_eq333_por_nivel.csv"), rej_header, rej_rows)
    write_csv(os.path.join(out_dir, "rmse_nowcast_por_nivel.csv"), ["level_ft", "rmse", "n_aircraft"],
              [[f"{a:.0f}", b, c] for a, b, c in per_level_rmse(nowcast, grid)])
    rows = [["nowcast", j, f"{grid.levels_ft[i]:.0f}", e] for j, d in nowcast.items() for i, e in d.items()]
    rows += [["operational", j, "mean", d["mean"]] for j, d in operational.items()]
    write_csv(os.path.join(out_dir, "erros_por_aeronave.csv"),
              ["evaluation", "aircraft", "level_ft", "error_kt"], rows)
    kt.save_config({**cfg, "baseline": params, "vis": VIS}, out_dir)


# =============================================================================
# TESTES DE SANIDADE
# =============================================================================
def _obs(t, h, y, ac="A"):
    return kt.Observation(int(math.ceil(t)), ac, ac, float(t), float(h), float(y))


def _run(obs_list, steps, x0=0.0, exclude=None):
    grid = kt.AltitudeGrid.from_config(kt.CONFIG)
    est, valid, rej = run_baseline(kt.group_by(obs_list, lambda o: o.step), 0, steps, grid, x0,
                                   baseline_params(), exclude)
    return est, valid, rej, grid


def _brute_force(obs_list, end, grid, x0, params):
    """Referência direta das Eqs. 3.31–3.34, recalculadas em TODOS os passos k = 0..end."""
    x = np.full(grid.n, x0)
    out = {}
    for k in range(end + 1):
        for i in range(grid.n):
            y = np.array([o.headwind_kt for o in obs_list
                          if k - params["window_s"] < o.t_raw <= k
                          and grid.nearest_level(o.h_ft) == i])
            if len(y):
                keep = np.abs(y - y.mean()) <= params["outlier_kt"]
                if keep.sum() >= params["min_obs"]:
                    x[i] = y[keep].mean()
        out[k] = x.copy()
    return out


def run_selftests():
    i1950 = 15                                                     # 450 + 15·100
    # 1. Antes de N ≥ 3: x0. Ao 3.º valor: média exata.
    o = [_obs(10, 1960, 10), _obs(20, 1990, 12), _obs(30, 1951, 14)]
    est, _, _, grid = _run(o, [25, 30], x0=4.0)
    assert grid.levels_ft[i1950] == 1950
    assert est[25][i1950] == 4.0 and abs(est[30][i1950] - 12.0) < 1e-12

    # 2. Fronteiras da Eq. 3.31: t_m = t_k entra; t_m = t_k − 600 sai. Depois, retenção.
    est, valid, _, _ = _run(o, [609, 610, 629, 630, 2000], x0=4.0)
    assert abs(est[609][i1950] - 12.0) < 1e-12 and valid[609][i1950]
    assert not valid[610][i1950] and abs(est[610][i1950] - 12.0) < 1e-12   # N = 2: retém
    assert abs(est[2000][i1950] - 12.0) < 1e-12 and not valid[2000][i1950]

    # 3. Eq. 3.33: [10,10,10,10,60] -> média 20; 60 desvia 40 kt (> 19.4) e sai; resultado 10.
    o = [_obs(t, 2000, y) for t, y in zip(range(1, 6), (10, 10, 10, 10, 60))]
    est, _, rej, _ = _run(o, [5])
    assert abs(est[5][i1950] - 10.0) < 1e-12
    assert len(rej) == 1 and rej[0].headwind_kt == 60.0

    # 4. Propriedade conhecida da regra pela média (masking): [0,0,60] -> média 20; os três
    #    desviam ≥ 20 kt > 19.4 kt, todos rejeitados, N = 0, retém x0.
    o = [_obs(t, 2000, y) for t, y in zip(range(1, 4), (0, 0, 60))]
    est, _, rej, _ = _run(o, [3], x0=4.0)
    assert est[3][i1950] == 4.0 and len(rej) == 3

    # 5. Leave-one-out: excluir a aeronave B remove o seu contributo.
    o = [_obs(t, 2000, 10, "A") for t in (1, 2, 3)] + [_obs(t, 2000, 30, "B") for t in (4, 5, 6)]
    est_all, _, _, _ = _run(o, [6])
    est_ex, _, _, _ = _run(o, [6], exclude="B")
    assert abs(est_all[6][i1950] - 20.0) < 1e-12 and abs(est_ex[6][i1950] - 10.0) < 1e-12

    # 6. Equivalência eventos vs passo a passo, com tráfego sintético e outliers.
    rng = np.random.default_rng(kt.CONFIG["seed"])
    o, t0 = [], 0.0
    while t0 < 1500:
        for n, h in enumerate(np.arange(3990, 410, -60.0)):
            y = 5 + 0.004 * h + rng.normal(0, 5.83) + (40 if rng.random() < 0.03 else 0)
            o.append(_obs(t0 + 4.8 * n + rng.random(), h, y, f"A{int(t0)}"))
        t0 += rng.uniform(60, 400)
    end = int(max(x.step for x in o)) + 700
    est, _, _, grid = _run(o, list(range(end + 1)), x0=4.0)
    ref = _brute_force(o, end, grid, 4.0, baseline_params())
    assert all(np.allclose(est[k], ref[k], atol=1e-12) for k in range(end + 1))

    # 7. Perfil constante sem ruído: o baseline recupera-o exatamente nos níveis com N ≥ 3.
    o = [_obs(5 * n + 100 * a, h, 5 + 0.004 * grid.levels_ft[grid.nearest_level(h)], f"A{a}")
         for a in range(3) for n, h in enumerate(np.arange(3990, 410, -60.0))]
    est, valid, _, _ = _run(o, [400])
    v = valid[400]
    assert v.sum() > 20 and np.allclose(est[400][v], 5 + 0.004 * grid.levels_ft[v])

    print("Testes de sanidade do baseline: todos passaram.")


# =============================================================================
# MAIN
# =============================================================================
def prepare(csv_path, metar_path):
    """Dados e configuração exatamente como em kalman_tuning_1julho.main()."""
    base = kt.CONFIG
    grid = kt.AltitudeGrid.from_config(base)
    metar = kt.load_metar_schedule(metar_path, base["scenario_date"])
    obs, n_rows, start_step = kt.load_observations(csv_path, metar, base)
    obs = kt.assign_passages(obs, base["passage_gap_s"])
    cfg = kt.derive_run_config(base, metar, start_step)
    end_step = max(o.step for o in obs)
    first = {j: min(o.step for o in v) for j, v in kt.group_by(obs, lambda o: o.aircraft).items()}
    test_ids = sorted(j for j, k in first.items() if k >= start_step + cfg["warmup_s"])
    print(f"Mensagens: {n_rows} | observações aceites: {len(obs)} | passagens: {len(first)} | "
          f"teste (pós warm-up): {len(test_ids)}")
    print(f"x0 = {cfg['x0_kt']:.2f} kt ({cfg['x0_source']})")
    return cfg, grid, obs, start_step, end_step, test_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="dados_1julho_12h16h.csv")
    ap.add_argument("--metar", default="METAR_julho.txt")
    ap.add_argument("--out", default="resultados_baseline_1julho")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-show", action="store_true", help="só guarda os PNG, não abre janelas")
    args = ap.parse_args()

    run_selftests()
    if args.selftest:
        return

    cfg, grid, obs, start_step, end_step, test_ids = prepare(args.csv, args.metar)
    params = baseline_params()
    obs_by_step = kt.group_by(obs, lambda o: o.step)

    nowcast = nowcast_errors(obs, obs_by_step, start_step, test_ids, grid, cfg["x0_kt"], params)
    operational = operational_errors(obs, obs_by_step, start_step, test_ids, grid, cfg["x0_kt"], params)
    metrics = {"nowcast": rmse_mae(nowcast), "operational": rmse_mae(operational)}
    print("\nBaseline, métricas (kt):")
    for kind, m in metrics.items():
        print(f"  {kind:12} RMSE {m['rmse']:5.2f} | MAE {m['mae']:5.2f} | n = {m['n']}")

    _, _, rejected = run_baseline(obs_by_step, start_step, [end_step], grid, cfg["x0_kt"], params)
    header, rows = rejections_by_level(obs, rejected, grid)
    print_table("Rejeições da Eq. 3.33 por nível, corrida completa do cenário:", header, rows)

    os.makedirs(args.out, exist_ok=True)
    save_outputs(metrics, header, rows, nowcast, operational, grid, cfg, params, args.out)
    plt = make_figures(cfg, grid, obs, start_step, end_step, params, nowcast, args.out)
    print(f"\nResultados, figuras e configuração guardados em: {args.out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
