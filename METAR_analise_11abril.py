import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def analyse_lppt_wind_combined(
    target_month, 
    start_day, 
    end_day, 
    file_name, 
    start_hour=6, 
    end_hour=23
):
    """
    Lê os dados METAR para LPPT, filtra pelo intervalo de dias pretendido e 
    pela janela horária configurada. Calcula o Headwind e o Absolute Crosswind 
    para a RWY 02 (025°) e traça um gráfico com a legenda posicionada no exterior.
    """
    # -------------------------------------------------------------
    # 1. Parâmetros operacionais e Preparação
    # -------------------------------------------------------------
    runway_bearing = 25  # Rumo magnético da RWY 02 de LPPT (025°)
    wind_data = []       # Lista que armazenará os dados validados

    if not os.path.exists(file_name):
        print(f"Error: File missing in directory: {file_name}")
        return

    # -------------------------------------------------------------
    # 2. Leitura e extração de dados do ficheiro METAR
    # -------------------------------------------------------------
    with open(file_name, "r") as file:
        for line in file:
            # Processa apenas registos de Lisboa que contenham velocidade de vento em nós (KT)
            if "LPPT" in line and "KT" in line:
                parts = line.split()
                date_str = parts[0]

                # Validação da estrutura temporal: 12 dígitos numéricos (YYYYMMDDHHMM)
                if len(date_str) == 12 and date_str.isdigit():
                    try:
                        dt = pd.to_datetime(date_str, format="%Y%m%d%H%M")
                    except ValueError:
                        continue

                    # Filtro de Mês e Dias
                    if dt.month == target_month and (start_day <= dt.day <= end_day):
                        
                        # Filtro Horário Preciso (start_hour:00 até end_hour:00 inclusive)
                        in_time_window = (
                            (start_hour <= dt.hour < end_hour) or 
                            (dt.hour == end_hour and dt.minute == 0)
                        )

                        if in_time_window:
                            wind_str = next((p for p in parts if p.endswith("KT") and len(p) >= 7), None)

                            if wind_str:
                                direction_str = wind_str[0:3]
                                speed_str = wind_str[3:5]

                                try:
                                    speed = int(speed_str)
                                except ValueError:
                                    speed = None

                                # -------------------------------------------------
                                # 3. Tratamento Trigonométrico
                                # -------------------------------------------------
                                # Ventos variáveis (VRB) ou inválidos são ignorados
                                if direction_str == "VRB" or speed is None:
                                    headwind = None
                                    crosswind = None
                                else:
                                    try:
                                        direction_deg = int(direction_str)
                                        # Conversão angular para radianos
                                        delta_rad = np.radians(direction_deg - runway_bearing)

                                        # Projeção das componentes de proa e través
                                        headwind = speed * np.cos(delta_rad)
                                        crosswind = speed * np.sin(delta_rad)
                                    except ValueError:
                                        headwind = None
                                        crosswind = None

                                # Armazenamento dos cálculos validados
                                wind_data.append({
                                    "DateTime": dt,
                                    "Wind Speed": speed,
                                    "Headwind": headwind,
                                    "Crosswind": crosswind,
                                })

    # -------------------------------------------------------------
    # 4. Estruturação no Pandas e Cálculo Absoluto
    # -------------------------------------------------------------
    df = pd.DataFrame(wind_data)

    if df.empty:
        print(
            f"No data found for Month {target_month}, Days {start_day}-{end_day} "
            f"between {start_hour:02d}:00 and {end_hour:02d}:00 UTC in '{file_name}'."
        )
        return

    # Limpeza de valores ausentes e ordenação cronológica
    df = df.dropna(subset=["Headwind", "Crosswind"])
    df = df.sort_values("DateTime").reset_index(drop=True)

    # Magnitude absoluta do vento de través para evitar cancelamentos
    df["Abs Crosswind"] = df["Crosswind"].abs()

    # -------------------------------------------------------------
    # 5. Cálculo das Médias
    # -------------------------------------------------------------
    mean_speed = df["Wind Speed"].mean()
    mean_headwind = df["Headwind"].mean()
    mean_abs_crosswind = df["Abs Crosswind"].mean()

    print(f"\n--- Wind Summary (Days {start_day} to {end_day} of Month {target_month}) [LPPT - RWY 02] ---")
    print(f"Time Window: {start_hour:02d}:00 to {end_hour:02d}:00 UTC")
    print(f"File processed: {file_name}")
    print(f"Mean Surface Wind Speed: {mean_speed:.2f} kt")
    print(f"Mean Net Headwind Component: {mean_headwind:.2f} kt")
    print(f"Mean Absolute Crosswind Component: {mean_abs_crosswind:.2f} kt")

    # -------------------------------------------------------------
    # 6. Geração do Gráfico (Legendas em Inglês Britânico)
    # -------------------------------------------------------------
    sns.set_theme(style="whitegrid")

    date_formatter = mdates.DateFormatter("%d %b %H:%M")
    major_locator = mdates.AutoDateLocator(maxticks=12)

    plt.figure(figsize=(15, 6))

    # Curva e média de Headwind
    sns.lineplot(
        data=df, x="DateTime", y="Headwind",
        marker="o", color="teal", linewidth=1.5, label="Instantaneous Net Headwind"
    )
    plt.axhline(
        mean_headwind, color="darkcyan", linestyle="--", linewidth=2,
        label=f"Mean Net Headwind ({mean_headwind:.1f} kt)"
    )

    # Curva e média de Absolute Crosswind
    sns.lineplot(
        data=df, x="DateTime", y="Abs Crosswind",
        marker="s", color="darkorange", linewidth=1.5, alpha=0.8, label="Instantaneous Absolute Crosswind"
    )
    plt.axhline(
        mean_abs_crosswind, color="chocolate", linestyle="--", linewidth=2,
        label=f"Mean Absolute Crosswind ({mean_abs_crosswind:.1f} kt)"
    )

    # Linha zero de referência
    plt.axhline(0, color="grey", linestyle=":", linewidth=1)

    # Nomes dos meses em inglês britânico
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
        7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
    }
    month_str = month_names.get(target_month, str(target_month))

    plt.title(
        f"Surface Wind Components from {start_day} to {end_day} {month_str} "
        f"({start_hour:02d}:00 - {end_hour:02d}:00 UTC) [LPPT - RWY 02]",
        fontsize=14, fontweight="bold"
    )
    plt.xlabel("Date & Time (UTC)", fontsize=12)
    plt.ylabel("Wind Component Speed (kt)", fontsize=12)

    plt.gca().xaxis.set_major_formatter(date_formatter)
    plt.gca().xaxis.set_major_locator(major_locator)
    plt.xticks(rotation=45)

    # =========================================================================
    # LEGENDA COLOCADA NO EXTERIOR DO GRÁFICO (À DIREITA)
    # =========================================================================
    plt.legend(
        loc="upper left",          # Fixa o canto superior esquerdo da caixa de legenda
        bbox_to_anchor=(1.02, 1),  # Desloca a caixa 2% à direita do limite do gráfico
        borderaxespad=0,           # Remove espaçamentos redundantes com o eixo
        frameon=True,              # Moldura visível
        facecolor="white",         # Fundo opaco para legibilidade
        framealpha=1.0,            # Fundo totalmente opaco
        edgecolor="silver",        # Bordo suave
        fontsize=10
    )

    # Ajusta os limites da figura para que a legenda externa não seja cortada
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ==============================================================================
    # ZONA DE CONFIGURAÇÃO DO UTILIZADOR
    # ==============================================================================
    MES_ALVO = 4                      # Mês pretendido (ex: 4 para Abril)
    DIA_INICIAL = 11                  # Dia inicial da análise
    DIA_FINAL = 11                    # Dia final da análise
    HORA_INICIAL = 16                  # Hora de início (ex: 08:00 UTC)
    HORA_FINAL = 20                   # Hora de fim (ex: 12:00 UTC -> bloco de 4 horas)
    NOME_FICHEIRO = "METAR_abril.txt" # Nome do ficheiro com os registos METAR

    # ==============================================================================
    # EXECUÇÃO DO PROGRAMA
    # ==============================================================================
    print("\nStarting wind analysis process...")
    analyse_lppt_wind_combined(
        target_month=MES_ALVO,
        start_day=DIA_INICIAL,
        end_day=DIA_FINAL,
        file_name=NOME_FICHEIRO,
        start_hour=HORA_INICIAL,
        end_hour=HORA_FINAL
    )