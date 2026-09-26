import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def analyse_lppt_july1_wind_afternoon():
    """Lê os dados METAR de 1 de Julho para LPPT, restringe a análise

    ao intervalo entre as 12:00 e as 16:00 UTC, calcula as componentes
    de Headwind e Crosswind para a Pista 02 (025°) e traça os gráficos
    com a informação do vento médio do período.
    """
    # -------------------------------------------------------------
    # 1. Parâmetros operacionais e configuração de ficheiros
    # -------------------------------------------------------------
    files = ["METAR_julho.txt"]
    runway_bearing = 25  # Rumo magnético da RWY 02 de LPPT (025°)[cite: 1]
    wind_data = []

    # -------------------------------------------------------------
    # 2. Leitura e extração com filtro temporal (12h - 16h)
    # -------------------------------------------------------------
    for file_name in files:
        if not os.path.exists(file_name):
            print(f"File missing in directory: {file_name}")
            continue

        with open(file_name, "r") as file:
            for line in file:
                if "LPPT" in line and "KT" in line:
                    parts = line.split()
                    date_str = parts[0]

                    if len(date_str) == 12 and date_str.isdigit():
                        try:
                            dt = pd.to_datetime(
                                date_str, format="%Y%m%d%H%M"
                            )
                        except ValueError:
                            continue

                        # Selecionar o dia 1 de Julho e o intervalo entre as 12h e as 16h (UTC)
                        if dt.month == 7 and dt.day == 1 and (12 <= dt.hour <= 16):
                            wind_str = next(
                                (
                                    p
                                    for p in parts
                                    if p.endswith("KT") and len(p) >= 7
                                ),
                                None,
                            )

                            if wind_str:
                                direction_str = wind_str[0:3]
                                speed_str = wind_str[3:5]

                                try:
                                    speed = int(speed_str)
                                except ValueError:
                                    speed = None

                                # -------------------------------------------------
                                # 3. Tratamento da direção e cálculo de componentes
                                # -------------------------------------------------
                                if (
                                    direction_str == "VRB"
                                    or speed is None
                                ):
                                    direction_deg = None
                                    headwind = None
                                    crosswind = None
                                else:
                                    try:
                                        direction_deg = int(direction_str)
                                        # Diferença angular convertida para radianos
                                        delta_rad = np.radians(
                                            direction_deg - runway_bearing
                                        )

                                        # Cálculo trigonométrico das componentes em nós (kt)
                                        headwind = speed * np.cos(delta_rad)
                                        crosswind = speed * np.sin(delta_rad)
                                    except ValueError:
                                        direction_deg = None
                                        headwind = None
                                        crosswind = None

                                wind_data.append(
                                    {
                                        "DateTime": dt,
                                        "Wind Speed": speed,
                                        "Wind Direction": direction_deg,
                                        "Headwind": headwind,
                                        "Crosswind": crosswind,
                                    }
                                )

    # -------------------------------------------------------------
    # 4. Estruturação e ordenação no Pandas
    # -------------------------------------------------------------
    df = pd.DataFrame(wind_data)

    if df.empty:
        print(
            "No data found for 1 July between 12:00 and 16:00 in the provided file."
        )
        return

    df = df.sort_values("DateTime").reset_index(drop=True)

    # -------------------------------------------------------------
    # 5. Cálculo das estatísticas do vento médio (12h - 16h)
    # -------------------------------------------------------------
    mean_speed = df["Wind Speed"].mean()
    mean_headwind = df["Headwind"].mean()
    mean_crosswind = df["Crosswind"].mean()

    print(
        "--- 1 July Wind Summary (12:00 - 16:00 UTC) [LPPT - RWY 02] ---"
    )
    print(f"Mean Surface Wind Speed: {mean_speed:.2f} kt")
    print(f"Mean Headwind Component: {mean_headwind:.2f} kt")
    print(f"Mean Crosswind Component: {mean_crosswind:.2f} kt")

    # -------------------------------------------------------------
    # 6. Configuração estética global e gráficos (Inglês Britânico)
    # -------------------------------------------------------------
    sns.set_theme(style="whitegrid")

    # --- GRÁFICO 1: Headwind Component ---
    plt.figure(figsize=(14, 5))
    sns.lineplot(
        data=df,
        x="DateTime",
        y="Headwind",
        marker="o",
        color="teal",
        linewidth=2,
        label="Instantaneous Headwind",
    )
    # Linha do Vento Médio de Proa no intervalo
    plt.axhline(
        mean_headwind,
        color="darkred",
        linestyle="--",
        linewidth=1.8,
        label=f"Period Mean Headwind ({mean_headwind:.1f} kt)",
    )
    # Linha de referência neutra (0 kt)
    plt.axhline(0, color="grey", linestyle=":", linewidth=1)

    plt.title(
        "Surface Headwind Component on 1 July (12:00 - 16:00 UTC) [LPPT - RWY 02]",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Time (UTC)", fontsize=12)
    plt.ylabel("Headwind Component (kt)", fontsize=12)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.gca().xaxis.set_major_locator(
        mdates.MinuteLocator(interval=30)
    )  # Localizador de 30 em 30 minutos
    plt.legend(
        loc="upper right", frameon=True, facecolor="white", framealpha=0.9
    )
    plt.tight_layout()
    plt.show()

    # --- GRÁFICO 2: Crosswind Component ---
    plt.figure(figsize=(14, 5))
    sns.lineplot(
        data=df,
        x="DateTime",
        y="Crosswind",
        marker="s",
        color="darkorange",
        linewidth=2,
        label="Instantaneous Crosswind",
    )
    # Linha do Vento Médio de Través no intervalo
    plt.axhline(
        mean_crosswind,
        color="navy",
        linestyle="--",
        linewidth=1.8,
        label=f"Period Mean Crosswind ({mean_crosswind:.1f} kt)",
    )
    # Linha de referência neutra (0 kt)
    plt.axhline(0, color="grey", linestyle=":", linewidth=1)

    plt.title(
        "Surface Crosswind Component on 1 July (12:00 - 16:00 UTC) [LPPT - RWY 02]",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Time (UTC)", fontsize=12)
    plt.ylabel("Crosswind Component (kt)", fontsize=12)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    plt.legend(
        loc="upper right", frameon=True, facecolor="white", framealpha=0.9
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyse_lppt_july1_wind_afternoon()