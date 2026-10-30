"""
Estilo comum das figuras da tese (cores e eixos), para que todas tenham o mesmo aspeto.
Legendas e rótulos das figuras em inglês britânico.
"""

INK = "#2b2b2b"          # texto e linhas de referência
INK_MUTED = "#6b6b6b"    # marcas e números dos eixos
GRID_COLOR = "#e4e4e2"
OBS_COLOR = "#2a78d6"    # observações das aeronaves

# Modelos: cores fixas por modelo em todas as figuras (rosa-choque, azul, verde-água; validadas
# para daltonismo em todos os pares e com contraste ≥ 3:1). O benchmark METAR é uma
# referência e mantém o tracejado.
MODEL_STYLE = {
    "kalman": {"color": "#ff00ff", "linestyle": "-"},
    "baseline": {"color": "#1c5cab", "linestyle": "-"},
    "metar": {"color": "#199e70", "linestyle": "--"},
}
MODEL_LINEWIDTH = 1.6

# Cap. 4: filtro com dois valores de q (q afinado = cor do filtro; q = 1 em roxo escuro,
# distinguível do rosa-choque também com deuteranopia).
KALMAN_Q_COLORS = {0.005: MODEL_STYLE["kalman"]["color"], 1.0: "#4b3bb5"}

# Observações individuais nos perfis (cinzento) e média observada por passagem nas séries.
OBS_POINTS = {"s": 18, "color": INK, "alpha": 0.55, "edgecolors": "none"}
OBS_MEAN_POINTS = {"s": 22, "color": INK, "edgecolors": "white", "linewidths": 0.6}
WARMUP_SPAN = {"color": GRID_COLOR, "alpha": 0.7, "zorder": 0}
TIME_TICK_S = 30 * 60

# Pontos das observações (nuvens de milhares de pontos)
OBS_SCATTER = {"s": 5, "color": OBS_COLOR, "alpha": 0.25, "edgecolors": "none"}

# Limites comuns dos eixos de altitude
ALTITUDE_LIM_FT = (0, 4400)


def style_axes(ax):
    """Grelha discreta por baixo dos dados, sem caixa superior/direita."""
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def panel_label(ax, text):
    """Identificação dentro do painel, no canto superior esquerdo (sem títulos)."""
    ax.text(0.03, 0.99, text, transform=ax.transAxes, va="top", ha="left", fontsize=9,
            color=INK, zorder=5, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})


def bottom_legend(fig, ax, ncol, bottom=0.12):
    """Legenda única por baixo da figura; devolve o retângulo para tight_layout."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, labelcolor=INK)
    return (0, bottom, 1, 1)


def format_time_axis(ax, tick_s=TIME_TICK_S):
    """Eixo x em HH:MM UTC, marcas a cada 30 min. Os instantes já estão em s desde 00:00 UTC
    ('tod' é UTC; alteração de 29-10-2026: deixou de haver conversão para hora local)."""
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    import preprocessamento as pp
    ax.xaxis.set_major_locator(MultipleLocator(tick_s))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda s, _pos: pp.fmt_hhmm(s)))
    ax.set_xlabel("Time (UTC)")
