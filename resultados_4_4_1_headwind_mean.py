"""
Visualização do Filtro de Kalman para dois valores de q — 1 de julho de 2026.

Gera duas figuras:
  1. Perfil vertical de headwind (eixo x) vs altitude (eixo y) num instante da
     simulação, para cada q, com todas as observações recebidas nos últimos 10 min.
  2. Headwind médio da aproximação ao longo do tempo, para cada q, com o headwind
     médio observado por cada aeronave como referência.
No fim, imprime as rejeições do gate (Eq. 3.20) por nível e no total, para cada q.

Reutiliza o pré-processamento, a conversão de tempo e o filtro de
'kalman_tuning_1julho.py', que tem de estar na mesma pasta. A configuração vem do
mesmo CONFIG, sem alterações, pelo que o filtro é o mesmo da afinação de q.

Uso:
  python kalman_visualizacao_q_1julho.py --csv dados_1julho_12h16h.csv --metar METAR_julho.txt
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

import estilo_figuras as ef
import resultados_4_4_1_q_tuning as kt

# =============================================================================
# CONFIGURAÇÃO DA VISUALIZAÇÃO
# =============================================================================
VIS = {
    "q_values": [0.005, 1.0],          # kt²/s; cores em estilo_figuras.KALMAN_Q_COLORS
    "snapshot_hour": 3.0,              # horas desde o início da simulação
    "mean_sample_s": 10,               # resolução temporal do gráfico do vento médio
    "recent_window_s": 10 * 60,        # janela das observações no perfil (= Eq. 3.31)
}

# =============================================================================
# DADOS E FILTRO
# =============================================================================
def prepare(csv_path, metar_path):
    """Carrega observações e deriva a configuração exatamente como no script de afinação."""
    base = kt.CONFIG
    grid = kt.AltitudeGrid.from_config(base)
    metar = kt.load_metar_schedule(metar_path, base["scenario_date"])
    obs, _, start_step = kt.load_observations(csv_path, metar, base)
    obs = kt.assign_passages(obs, base["passage_gap_s"])
    cfg = kt.derive_run_config(base, metar, start_step)
    end_step = max(o.step for o in obs)
    print(f"Observações aceites: {len(obs)} | x0 = {cfg['x0_kt']:.2f} kt ({cfg['x0_source']}) | "
          f"P0 = {cfg['p0_kt2']:.1f} kt²")
    return cfg, obs, grid, start_step, end_step


def run_for_q(cfg, obs, grid, start_step, end_step, snapshot_step, q):
    """Corre o filtro e devolve o perfil no instante pedido, a série do vento médio e
    as rejeições do gate por nível."""
    obs_by_step = kt.group_by(obs, lambda o: o.step)
    t_mean = list(range(start_step, end_step + 1, VIS["mean_sample_s"]))
    est, gated_by_level = kt.run_filter(obs_by_step, start_step, t_mean + [snapshot_step],
                                        grid, q, cfg)
    # Média simples dos 36 níveis (Eq. 3.30 sem o termo de superfície)
    mean_hw = np.array([est[k].mean() for k in t_mean])
    return est[snapshot_step], np.array(t_mean), mean_hw, gated_by_level


def observed_mean_per_aircraft(obs, grid):
    """Headwind médio observado por cada aeronave (média dos níveis, Eqs. 3.35 e 3.37),
    colocado no instante da sua primeira observação aceite."""
    t, v = [], []
    for ac_obs in kt.group_by(obs, lambda o: o.aircraft).values():
        ref, _ = kt.reference_by_level(ac_obs, grid)
        t.append(min(o.step for o in ac_obs))
        v.append(np.mean(list(ref.values())))
    return np.array(t), np.array(v)


# =============================================================================
# FIGURAS
# =============================================================================
def recent_observations(obs, snapshot_step, window_s):
    """Todas as observações recebidas na janela t_k - window < t_m ≤ t_k (mesma janela
    da Eq. 3.31 do baseline), sem rejeição de outliers."""
    return [o for o in obs if snapshot_step - window_s < o.t_raw <= snapshot_step]


def kalman_label(q):
    return f"Kalman filter, q = {q:g} kt²/s"


def plot_profile(cfg, grid, profiles, obs, snapshot_step):
    """Estilo comum (estilo_figuras.py): sem título, instante dentro do painel, legenda em baixo."""
    window_s = VIS["recent_window_s"]
    recent = recent_observations(obs, snapshot_step, window_s)

    fig, ax = plt.subplots(figsize=(7.5, 8))
    ax.scatter([o.headwind_kt for o in recent], [o.h_ft for o in recent], zorder=2,
               label=f"Observations in the last {window_s // 60:g} min", **ef.OBS_POINTS)
    for q, x in profiles.items():
        ax.plot(x, grid.levels_ft, color=ef.KALMAN_Q_COLORS[q], linewidth=ef.MODEL_LINEWIDTH,
                zorder=3, label=kalman_label(q))
    ef.panel_label(ax, f"{kt.fmt_hhmm(snapshot_step)} UTC (T = {VIS['snapshot_hour']:g} h)")
    ax.set_xlabel("Headwind [kt]")
    ax.set_ylabel("True altitude [ft]")
    ax.set_ylim(*ef.ALTITUDE_LIM_FT)
    ef.style_axes(ax)
    fig.tight_layout(rect=ef.bottom_legend(fig, ax, ncol=1, bottom=0.1))
    return fig


def plot_mean_series(cfg, series, obs, grid, start_step, end_step):
    t_ac, v_ac = observed_mean_per_aircraft(obs, grid)

    fig, ax = plt.subplots(figsize=(11, 5))
    t0 = start_step
    ax.axvspan(t0, t0 + cfg["warmup_s"], label=f"Warm-up period ({cfg['warmup_s'] // 60:g} min)",
               **ef.WARMUP_SPAN)
    for q, (t, mean_hw) in series.items():
        ax.plot(t, mean_hw, color=ef.KALMAN_Q_COLORS[q],
                linewidth=ef.MODEL_LINEWIDTH, zorder=3, label=kalman_label(q))
    ax.scatter(t_ac, v_ac, zorder=4,
               label="Mean headwind observed by each passage", **ef.OBS_MEAN_POINTS)
    ef.format_time_axis(ax)
    ax.set_ylabel("Mean headwind of the 36 levels [kt]")
    ef.style_axes(ax)
    fig.tight_layout(rect=ef.bottom_legend(fig, ax, ncol=4, bottom=0.08))
    return fig


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="dados_1julho_12h16h.csv")
    ap.add_argument("--metar", default="METAR_julho.txt")
    ap.add_argument("--out", default="resultados_visualizacao_1julho")
    ap.add_argument("--no-show", action="store_true", help="só guarda os PNG, não abre janelas")
    args = ap.parse_args()

    cfg, obs, grid, start_step, end_step = prepare(args.csv, args.metar)
    snapshot_step = start_step + int(VIS["snapshot_hour"] * 3600)
    if snapshot_step > end_step:
        raise ValueError("O instante do perfil está depois do fim dos dados.")

    profiles, series, gated_by_q = {}, {}, {}
    for q in VIS["q_values"]:
        profile, t, mean_hw, gated_by_q[q] = run_for_q(cfg, obs, grid, start_step, end_step,
                                                       snapshot_step, q)
        profiles[q], series[q] = profile, (t, mean_hw)

    os.makedirs(args.out, exist_ok=True)
    fig1 = plot_profile(cfg, grid, profiles, obs, snapshot_step)
    fig1.savefig(os.path.join(args.out, "perfil_headwind_T3h.png"), dpi=150)
    fig2 = plot_mean_series(cfg, series, obs, grid, start_step, end_step)
    fig2.savefig(os.path.join(args.out, "headwind_medio_tempo.png"), dpi=150)
    kt.save_config({**cfg, "vis": VIS}, args.out)
    print(f"Figuras e configuração guardadas em: {args.out}")

    kt.report_gate_rejections(obs, grid, gated_by_q,
                              os.path.join(args.out, "rejeicoes_gate_1julho.csv"))
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
