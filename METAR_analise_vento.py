"""
Análise do vento à superfície em LPPT a partir de relatórios METAR (1 Mar – 31 Jul 2026).

Junta METAR_analise3.py (Figura 4.3) e METAR_analise_crosswind.py (Figura 4.4) num só
script, com leitura dos ficheiros feita uma única vez.

Figuras produzidas (legendas em inglês britânico):
  - Figura 4.3: velocidade média/mín./máx. diária e categorização da direção predominante.
  - Figura 4.4: componentes diárias de headwind e |crosswind| para a pista 02.

Referências na tese:
  - Secção 3.1 (Tabela 3.2, formato METAR dddssKT; direção em graus a partir do norte verdadeiro).
  - Eqs. (3.1) e (3.2): V_HW = V cos(Δθ), V_CW = V sin(Δθ), Δθ = θ_w − θ_RWY, θ_RWY = 22.72°.
  - Secção 4.2.1: período operacional 06:00–23:59 UTC.

Utilização:
  python METAR_analise_vento.py                      # lê os METAR da pasta atual, guarda e mostra as figuras
  python METAR_analise_vento.py --data-dir pasta --no-show

Resultados: pasta METAR_analise_vento/ ao lado do script, com
  METAR_analise_vento_fig4_3.png, METAR_analise_vento_fig4_4.png e METAR_analise_vento.txt
  python METAR_analise_vento.py --self-test          # testes de sanidade, sem precisar dos ficheiros
"""

import argparse
import os
import re

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# =============================================================================
# CONFIGURAÇÃO CENTRALIZADA
# =============================================================================
METAR_FILES = [
    "METAR_março.txt",
    "METAR_abril.txt",
    "METAR_maio.txt",
    "METAR_junho.txt",
    "METAR_julho.txt",
]

STATION = "LPPT"

# Horas excluídas (UTC): 00:00–05:59. Mantém-se o período 06:00–23:59 da Secção 4.2.1.
FIRST_OPERATIONAL_HOUR = 6

# Rumo verdadeiro da pista 02 (Secção 3.1, texto antes das Eqs. 3.1–3.2).
# NOTA: os scripts antigos usavam 25°. Para reproduzir exatamente a Figura 4.4 antiga, pôr 25.0.
RUNWAY_02_TRUE_BEARING_DEG = 22.72

# Incluir relatórios SPECI (os scripts antigos incluíam-nos implicitamente).
INCLUDE_SPECI = True

CAT_NNE = "North-Northeast Predominant Wind"
CAT_SSW = "South-Southwest Predominant Wind"
CAT_UNKNOWN = "Unknown"

# --- Cores ---
# Figura 4.3: verde escuro moderado (NNE) e vermelho-salmão suave (SSW).
# Diferem bastante em luminosidade para continuarem distinguíveis com daltonismo
# vermelho-verde (validado: ΔE OKLab 15.3 em protanopia; 29.0 em visão normal).
COLOUR_NNE = "#2E7D4F"
COLOUR_SSW = "#E8958A"
MARKER_NNE = "o"
MARKER_SSW = "s"          # codificação secundária: forma diferente além da cor
COLOUR_DAILY_SPEED = "grey"
COLOUR_OVERALL_MEAN = "purple"

# Figura 4.4: as mesmas cores das Figuras 4.5–4.8 ("surface wind components" da tese),
# confirmadas pelos píxeis das figuras: teal (#008080) e darkorange (#FF8C00).
COLOUR_HEADWIND = "#008080"
COLOUR_CROSSWIND = "#FF8C00"

SHOW_TITLES = False       # as figuras na tese não têm título (a legenda vai no caption LaTeX)
FIG_DPI = 300

# Resultados (figuras + resumo em texto) guardados numa pasta com o nome deste script,
# ao lado do próprio script: METAR_analise_vento/METAR_analise_vento_*.png e .txt
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), SCRIPT_NAME)

# Expressão regular do grupo de vento: dddff(f)(Gfmfm)KT ou VRBff(f)(Gfmfm)KT
WIND_GROUP_RE = re.compile(r"^(?P<dir>\d{3}|VRB)(?P<speed>\d{2,3})(?:G\d{2,3})?KT$")


# =============================================================================
# LEITURA E PARSING
# =============================================================================
def parse_metar_line(line):
    """
    Interpreta uma linha do ficheiro METAR (formato Ogimet: 'YYYYMMDDhhmm METAR LPPT ...').

    Devolve um dicionário com timestamp, direção (None se VRB), velocidade (kt) e tipo,
    ou None se a linha não tiver um grupo de vento válido. As rajadas (Gxx) são ignoradas.
    """
    parts = line.split()
    if len(parts) < 3 or STATION not in parts:
        return None

    stamp = parts[0]
    if len(stamp) != 12 or not stamp.isdigit():
        return None

    report_type = parts[1]  # "METAR" ou "SPECI"
    if report_type == "SPECI" and not INCLUDE_SPECI:
        return None

    wind_match = None
    for token in parts[2:]:
        wind_match = WIND_GROUP_RE.match(token)
        if wind_match:
            break
    if wind_match is None:  # p.ex. "METAR LPPT NIL="
        return None

    direction_str = wind_match.group("dir")
    direction = None if direction_str == "VRB" else int(direction_str)

    return {
        "Timestamp": pd.to_datetime(stamp, format="%Y%m%d%H%M"),
        "Direction": direction,
        "Wind Speed": int(wind_match.group("speed")),
        "Report Type": report_type,
    }


def load_metar_files(file_names, data_dir="."):
    """Lê todos os ficheiros METAR e devolve um DataFrame com uma linha por relatório."""
    records = []
    for file_name in file_names:
        path = os.path.join(data_dir, file_name)
        if not os.path.exists(path):
            print(f"File missing in directory: {path}")
            continue
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                record = parse_metar_line(line)
                if record is not None:
                    records.append(record)
    return pd.DataFrame(records)


def keep_operational_hours(df, first_hour=FIRST_OPERATIONAL_HOUR):
    """Remove os relatórios entre 00:00 e first_hour−1:59 UTC."""
    return df[df["Timestamp"].dt.hour >= first_hour].copy()


# =============================================================================
# CLASSIFICAÇÃO E COMPONENTES DO VENTO
# =============================================================================
def angle_to_runway_rad(direction_deg, runway_bearing_deg=RUNWAY_02_TRUE_BEARING_DEG):
    """Δθ = θ_w − θ_RWY em radianos (Secção 3.1)."""
    return np.radians(np.asarray(direction_deg, dtype=float) - runway_bearing_deg)


def classify_direction(direction_deg, runway_bearing_deg=RUNWAY_02_TRUE_BEARING_DEG):
    """
    NNE se o vento tem componente de frente na pista 02 (cos Δθ > 0); SSW caso contrário.

    Equivale à regra antiga 115° ≤ θ ≤ 295° → SSW para direções METAR (múltiplos de 10°),
    tanto com 22.72° como com 25° (ver self-test). VRB → Unknown.
    """
    if direction_deg is None or pd.isna(direction_deg):
        return CAT_UNKNOWN
    return CAT_NNE if np.cos(angle_to_runway_rad(direction_deg, runway_bearing_deg)) > 0 else CAT_SSW


def add_wind_components(df, runway_bearing_deg=RUNWAY_02_TRUE_BEARING_DEG):
    """Acrescenta Headwind (Eq. 3.1) e |Crosswind| (|Eq. 3.2|) em kt."""
    df = df.copy()
    delta = angle_to_runway_rad(df["Direction"], runway_bearing_deg)
    df["Headwind"] = df["Wind Speed"] * np.cos(delta)
    df["Crosswind"] = np.abs(df["Wind Speed"] * np.sin(delta))
    return df


# =============================================================================
# AGREGAÇÃO DIÁRIA
# =============================================================================
def daily_speed_summary(df):
    """Média/mín./máx. diária da velocidade e categoria mais frequente (moda) por dia."""
    df = df.assign(Date=df["Timestamp"].dt.normalize())
    return (
        df.groupby("Date")
        .agg(
            Mean_Speed=("Wind Speed", "mean"),
            Min_Speed=("Wind Speed", "min"),
            Max_Speed=("Wind Speed", "max"),
            # Em caso de empate, mode()[0] devolve a primeira por ordem alfabética (NNE).
            Category=("Category", lambda x: x.mode()[0]),
        )
        .reset_index()
    )


def daily_component_summary(df):
    """Média/mín./máx. diária do headwind e do |crosswind|."""
    df = df.assign(Date=df["Timestamp"].dt.normalize())
    return (
        df.groupby("Date")
        .agg(
            Mean_HW=("Headwind", "mean"),
            Min_HW=("Headwind", "min"),
            Max_HW=("Headwind", "max"),
            Mean_CW=("Crosswind", "mean"),
            Min_CW=("Crosswind", "min"),
            Max_CW=("Crosswind", "max"),
        )
        .reset_index()
    )


# =============================================================================
# GRÁFICOS
# =============================================================================
def format_date_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.setp(ax.get_xticklabels(), rotation=45)


def plot_speed_and_direction(df_daily, overall_mean_speed):
    """Figura 4.3."""
    fig, ax = plt.subplots(figsize=(16, 6))

    ax.fill_between(
        df_daily["Date"], df_daily["Min_Speed"], df_daily["Max_Speed"],
        color=COLOUR_DAILY_SPEED, alpha=0.2, label="Daily Min-Max Wind Speed Range", zorder=1,
    )
    ax.plot(
        df_daily["Date"], df_daily["Mean_Speed"],
        color=COLOUR_DAILY_SPEED, alpha=0.6, linewidth=1.5, zorder=2,
        label="Daily Average Surface Wind Speed",
    )
    ax.axhline(
        overall_mean_speed, color=COLOUR_OVERALL_MEAN, linestyle="--", linewidth=2, zorder=3,
        label=f"Overall Mean Surface Wind ({overall_mean_speed:.1f} kt)",
    )

    for category, colour, marker in [
        (CAT_SSW, COLOUR_SSW, MARKER_SSW),
        (CAT_NNE, COLOUR_NNE, MARKER_NNE),
    ]:
        subset = df_daily[df_daily["Category"] == category]
        ax.scatter(
            subset["Date"], subset["Mean_Speed"],
            color=colour, marker=marker, s=70, edgecolors="white", linewidths=0.8,
            zorder=4, label=category,
        )

    if SHOW_TITLES:
        ax.set_title(
            "Daily Average, Minimum, and Maximum Surface Wind Speed and Wind"
            " Direction Categorisation (06:00 - 24:00 UTC)",
            fontsize=15, fontweight="bold",
        )
    ax.set_xlabel("Date (Continuous Timeline)", fontsize=12)
    ax.set_ylabel("Wind Speed (kt)", fontsize=12)
    format_date_axis(ax)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_headwind_crosswind(df_daily):
    """
    Figura 4.4.

    Estilo alinhado com a Figura 4.3: mesmo tamanho, linhas finas, marcadores pequenos e
    bandas muito transparentes sem contorno (a sobreposição de duas bandas opacas criava
    uma terceira cor acastanhada que não corresponde a nenhuma série).
    """
    fig, ax = plt.subplots(figsize=(16, 6))

    series = [
        ("HW", COLOUR_HEADWIND, "o", "Headwind"),
        ("CW", COLOUR_CROSSWIND, "s", "Crosswind"),
    ]
    for key, colour, marker, name in series:
        ax.fill_between(
            df_daily["Date"], df_daily[f"Min_{key}"], df_daily[f"Max_{key}"],
            color=colour, alpha=0.10, linewidth=0, label=f"Daily {name} Range", zorder=1,
        )
    for key, colour, marker, name in series:
        ax.plot(
            df_daily["Date"], df_daily[f"Mean_{key}"],
            color=colour, linewidth=1.3, marker=marker, markersize=5.5,
            markeredgecolor="white", markeredgewidth=0.7, zorder=3,
            label=f"Daily Average {name} Component",
        )

    if SHOW_TITLES:
        ax.set_title(
            "Daily Average and Variation of Headwind and Crosswind Components (Runway 02 Only)",
            fontsize=15, fontweight="bold",
        )
    ax.set_xlabel("Date (Continuous Timeline)", fontsize=12)
    ax.set_ylabel("Wind Component Speed (kt)", fontsize=12)
    ax.set_ylim(bottom=0)
    format_date_axis(ax)
    # Ordem da legenda: média e intervalo de cada componente lado a lado.
    handles, labels = ax.get_legend_handles_labels()
    order = [2, 0, 3, 1]
    ax.legend(
        [handles[i] for i in order], [labels[i] for i in order],
        loc="upper right", frameon=True, facecolor="white", framealpha=0.9,
    )
    fig.tight_layout()
    return fig


# =============================================================================
# TESTES DE SANIDADE
# =============================================================================
def run_self_tests():
    """Casos com solução conhecida. Não precisa dos ficheiros METAR."""
    # 1) Parsing de linhas reais (METAR_março.txt), incluindo rajada, VRB, calmaria, COR e NIL.
    r = parse_metar_line("202603061630 METAR LPPT 061630Z 34022G34KT 9999 FEW026 12/06 Q1017 WS R02=")
    assert r["Direction"] == 340 and r["Wind Speed"] == 22, "rajada mal interpretada"
    r = parse_metar_line("202603311230 METAR LPPT 311230Z VRB09KT CAVOK 23/07 Q1026=")
    assert r["Direction"] is None and r["Wind Speed"] == 9, "VRB mal interpretado"
    r = parse_metar_line("202603160300 METAR LPPT 160300Z 00000KT CAVOK 11/09 Q1020=")
    assert r["Direction"] == 0 and r["Wind Speed"] == 0, "calmaria mal interpretada"
    r = parse_metar_line("202603181800 METAR COR LPPT 181800Z 15010KT 120V180 9999 FEW020 16/12 Q1001=")
    assert r["Direction"] == 150 and r["Wind Speed"] == 10, "METAR COR mal interpretado"
    assert parse_metar_line("202603030400 METAR LPPT NIL=") is None, "NIL deveria ser ignorado"
    # A variação de direção (120V180) não pode ser confundida com o grupo de vento.
    r = parse_metar_line("202603011330 METAR LPPT 011330Z 06005KT 340V110 9999 FEW025 16/09 Q1020=")
    assert r["Direction"] == 60 and r["Wind Speed"] == 5

    # 2) Equivalência com a regra antiga (115 ≤ θ ≤ 295 → SSW) para direções METAR.
    for bearing in (22.72, 25.0):
        for d in range(0, 361, 10):
            old = CAT_SSW if 115 <= d <= 295 else CAT_NNE
            assert classify_direction(d, bearing) == old, f"classificação difere em {d}° (rumo {bearing})"
    assert classify_direction(None) == CAT_UNKNOWN

    # 3) Componentes com solução conhecida (Eqs. 3.1–3.2).
    b = RUNWAY_02_TRUE_BEARING_DEG
    test = pd.DataFrame({"Direction": [b, b + 90, b + 180, b - 90, b + 60], "Wind Speed": [10] * 5})
    comp = add_wind_components(test)
    expected_hw = [10, 0, -10, 0, 5]
    expected_cw = [0, 10, 0, 10, 10 * np.sin(np.radians(60))]
    assert np.allclose(comp["Headwind"], expected_hw, atol=1e-9), comp["Headwind"].tolist()
    assert np.allclose(comp["Crosswind"], expected_cw, atol=1e-9), comp["Crosswind"].tolist()
    # Conservação: HW² + CW² = V²
    assert np.allclose(comp["Headwind"] ** 2 + comp["Crosswind"] ** 2, 100)

    # 4) Filtro horário: 05:30 excluído, 06:00 incluído.
    hours = pd.DataFrame({"Timestamp": pd.to_datetime(["2026-03-01 05:30", "2026-03-01 06:00"])})
    assert len(keep_operational_hours(hours)) == 1

    print("Self-tests: all passed.")


# =============================================================================
# PRINCIPAL
# =============================================================================
def analyse_lppt_wind(data_dir=".", show=True):
    sns.set_theme(style="whitegrid")
    report_lines = []

    def report(text=""):
        """Escreve na consola e guarda a linha para o ficheiro de resultados."""
        print(text)
        report_lines.append(str(text))

    df = load_metar_files(METAR_FILES, data_dir)
    if df.empty:
        print("No METAR data found.")
        return
    df = keep_operational_hours(df)
    df["Category"] = df["Direction"].apply(classify_direction)

    # Média global: calculada ANTES de remover VRB, como no script original (dá os 9.1 kt da tese).
    overall_mean_speed = df["Wind Speed"].mean()
    report("LPPT METAR surface wind analysis (06:00-23:59 UTC)")
    report(f"Runway 02 true bearing: {RUNWAY_02_TRUE_BEARING_DEG} deg | SPECI included: {INCLUDE_SPECI}")
    report(f"Period: {df['Timestamp'].min():%d %b %Y} to {df['Timestamp'].max():%d %b %Y}")
    report(f"Reports used: {len(df)} (VRB: {(df['Category'] == CAT_UNKNOWN).sum()})")
    report(f"Overall Surface Mean Wind Speed: {overall_mean_speed:.2f} kt")

    df_known = df[df["Category"] != CAT_UNKNOWN].copy()

    # --- Figura 4.3 ---
    df_daily_speed = daily_speed_summary(df_known)
    report()
    report("Days by predominant wind direction (Figure 4.3):")
    report(df_daily_speed["Category"].value_counts().to_string())
    fig_speed = plot_speed_and_direction(df_daily_speed, overall_mean_speed)

    # --- Figura 4.4 --- (filtro por observação, como no script original)
    df_nne = add_wind_components(df_known[df_known["Category"] == CAT_NNE])
    df_daily_comp = daily_component_summary(df_nne)
    report()
    report("Runway 02 components, NNE observations only (Figure 4.4):")
    report(f"  Mean headwind: {df_nne['Headwind'].mean():.2f} kt | Mean |crosswind|: {df_nne['Crosswind'].mean():.2f} kt")
    fig_comp = plot_headwind_crosswind(df_daily_comp)

    # --- Guardar resultados ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path_fig43 = os.path.join(OUTPUT_DIR, f"{SCRIPT_NAME}_fig4_3.png")
    path_fig44 = os.path.join(OUTPUT_DIR, f"{SCRIPT_NAME}_fig4_4.png")
    path_txt = os.path.join(OUTPUT_DIR, f"{SCRIPT_NAME}.txt")
    fig_speed.savefig(path_fig43, dpi=FIG_DPI)
    fig_comp.savefig(path_fig44, dpi=FIG_DPI)
    with open(path_txt, "w", encoding="utf-8") as file:
        file.write("\n".join(report_lines) + "\n")
    print(f"\nResults saved to: {OUTPUT_DIR}")

    if show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPPT METAR surface wind analysis.")
    parser.add_argument("--data-dir", default=".", help="Pasta com os ficheiros METAR_*.txt")
    parser.add_argument("--no-show", action="store_true", help="Não abrir as janelas das figuras")
    parser.add_argument("--self-test", action="store_true", help="Correr apenas os testes de sanidade")
    args = parser.parse_args()

    if args.self_test:
        run_self_tests()
    else:
        analyse_lppt_wind(data_dir=args.data_dir, show=not args.no_show)
