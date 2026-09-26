import os
import math
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def analyse_lppt_wind():
    # 1. Lista com os nomes dos ficheiros METAR
    files = [
        "METAR_março.txt",
        "METAR_abril.txt",
        "METAR_maio.txt",
        "METAR_junho.txt",
        "METAR_julho.txt",
    ]
    wind_data = []

    # 2. Leitura sequencial dos ficheiros de texto
    for file_name in files:
        if not os.path.exists(file_name):
            print(f"File missing in directory: {file_name}")
            continue

        with open(file_name, "r") as file:
            for line in file:
                # 3. Filtrar linhas de observação do LPPT com dados de vento (KT)
                if "LPPT" in line and "KT" in line:
                    parts = line.split()
                    date_str = parts[0]

                    if len(date_str) == 12 and date_str.isdigit():
                        date_formatted = date_str[0:8]  # Formato YYYYMMDD
                        hour_str = date_str[8:10]       # Formato HH

                        try:
                            hour = int(hour_str)
                        except ValueError:
                            continue

                        # Descartar dados da meia-noite até às 06:00
                        if 0 <= hour < 6:
                            continue

                        wind_str = next(
                            (p for p in parts if p.endswith("KT") and len(p) >= 7), None
                        )

                        if wind_str:
                            direction_str = wind_str[0:3]
                            speed_str = wind_str[3:5]

                            try:
                                speed = int(speed_str)
                            except ValueError:
                                continue

                            # 4. Classificação operacional do vento e obtenção do ângulo
                            direction = None
                            if direction_str == "VRB":
                                runway = "Unknown"
                            else:
                                try:
                                    direction = int(direction_str)
                                    if 115 <= direction <= 295:
                                        runway = "South-Southwest Predominant Wind"
                                    else:
                                        runway = "North-Northeast Predominant Wind"
                                except ValueError:
                                    runway = "Unknown"

                            wind_data.append({
                                "Date": date_formatted,
                                "Direction": direction,
                                "Wind Speed": speed,
                                "Runway": runway,
                            })

    # 5. Estruturação no Pandas
    df = pd.DataFrame(wind_data)

    if df.empty:
        print("No data available.")
        return

    # 6. FILTRO: Excluir os pontos laranjas (Focar apenas na Pista 02)
    df = df[df["Runway"] == "North-Northeast Predominant Wind"].copy()
    
    # Garantir o formato de data
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")

    # 7. Cálculo das componentes (Headwind e Crosswind)
    # A orientação da Pista 02 em Lisboa é de exatos 25 graus.
    runway_heading_02 = 25 
    
    # O NumPy precisa do ângulo em radianos para aplicar seno e cosseno
    # Subtraímos a direção do vento ao ângulo da pista
    df["Angle_Diff_Rad"] = np.radians(df["Direction"] - runway_heading_02)
    
    # Headwind: Multiplicamos a velocidade pelo cosseno do ângulo
    df["Headwind"] = df["Wind Speed"] * np.cos(df["Angle_Diff_Rad"])
    
    # Crosswind: Multiplicamos a velocidade pelo seno do ângulo. 
    # Aplicamos .abs() para nos focar apenas na intensidade e não no lado da pista.
    df["Crosswind"] = (df["Wind Speed"] * np.sin(df["Angle_Diff_Rad"])).abs()

    # 8. Agrupamento diário avançado (.agg para Média, Mínimo, Máximo das novas variáveis)
    df_daily = (
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

    # 9. Renderização do Gráfico com Headwind e Crosswind (Legendas em Inglês Britânico)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(16, 8))

    # --- DESENHO DO HEADWIND (Tons de Azul) ---
    plt.fill_between(
        df_daily["Date"],
        df_daily["Min_HW"],
        df_daily["Max_HW"],
        color="royalblue",
        alpha=0.15,
        label="Daily Headwind Range",
        zorder=1,
    )
    sns.lineplot(
        data=df_daily,
        x="Date",
        y="Mean_HW",
        color="royalblue",
        marker="o",
        markersize=8,
        linewidth=2,
        zorder=2,
        label="Daily Average Headwind Component",
    )

    # --- DESENHO DO CROSSWIND (Tons de Verde) ---
    plt.fill_between(
        df_daily["Date"],
        df_daily["Min_CW"],
        df_daily["Max_CW"],
        color="forestgreen",
        alpha=0.15,
        label="Daily Crosswind Range",
        zorder=1,
    )
    sns.lineplot(
        data=df_daily,
        x="Date",
        y="Mean_CW",
        color="forestgreen",
        marker="s", # Marcadores quadrados para diferenciar do headwind
        markersize=8,
        linewidth=2,
        zorder=2,
        label="Daily Average Crosswind Component",
    )

    # Rótulos em Inglês Britânico
    plt.title(
        "Daily Average and Variation of Headwind and Crosswind Components (Runway 02 Only)",
        fontsize=15,
        fontweight="bold",
    )
    plt.xlabel("Date (Continuous Timeline)", fontsize=12)
    plt.ylabel("Wind Component Speed (kt)", fontsize=12)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=45)

    # Legenda posicionada no canto superior direito
    plt.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyse_lppt_wind()