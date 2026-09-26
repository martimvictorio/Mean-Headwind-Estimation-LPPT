import csv
import math
import re
import os

# ==========================================
# THESIS CONSTANTS
# ==========================================
X_THR = -433.76  
Y_THR = 277.06   
M_PARA_NM = 1852.0  
FT_PARA_M = 0.3048  # Conversion factor (feet to metres)

# ==========================================
# STANDARD ATMOSPHERE (ISA) CONSTANTS
# ==========================================
P_ISA = 1013.25      # Sea level pressure (hPa)
T_ISA = 288.15       # Sea level temperature (K)
LAMBDA_ISA = -0.0065 # Tropospheric temperature lapse rate (K/m)
R_GAS = 287.05287    # Specific gas constant for dry air (J/(kg·K))
G0 = 9.80665         # Gravitational acceleration (m/s²)


def carregar_cronograma_metar_em_segundos(caminho_metar, dia_alvo):
    """
    Reads the METAR file and creates a chronological dictionary for the specified day.
    Keys are converted to seconds of the day to easily sync with the simulation clock.
    """
    cronograma = {}
    if not os.path.exists(caminho_metar):
        print(f"[WARNING] File {caminho_metar} not found. Default ISA weather will be used.")
        return cronograma
        
    with open(caminho_metar, 'r', encoding='utf-8') as f:
        for linha in f:
            # Matches pattern like 011400Z (Day 01, Time 14:00 Zulu)
            padrao = rf'\b{dia_alvo}(\d{{2}})(\d{{2}})Z\b'
            match = re.search(padrao, linha)
            if match:
                hora = int(match.group(1))
                minuto = int(match.group(2))
                # Convert METAR time into total seconds since the start of the day
                tempo_em_segundos = (hora * 3600) + (minuto * 60)
                cronograma[tempo_em_segundos] = linha.strip()
                
    return dict(sorted(cronograma.items()))


def process_metar(metar_str):
    """
    Parses the METAR string to extract local QNH and Temperature.
    """
    # Extract QNH
    match_qnh = re.search(r'Q(\d{4})', metar_str)
    qnh = float(match_qnh.group(1)) if match_qnh else P_ISA
    
    # Extract Temperature
    match_temp = re.search(r'\b(M?\d{2})/(M?\d{2})\b', metar_str)
    if match_temp:
        temp_str = match_temp.group(1)
        temp_celsius = float(temp_str.replace('M', '-'))
    else:
        temp_celsius = 15.0 # Standard atmosphere fallback

    return qnh, temp_celsius


def convert_to_seconds(tod_str):
    """
    Converts ASTERIX timestamp string to total seconds.
    """
    partes = str(tod_str).split(':')
    if len(partes) == 3:
        return (float(partes[0]) * 3600) + (float(partes[1]) * 60) + float(partes[2])
    elif len(partes) == 2:
        return (float(partes[0]) * 60) + float(partes[1])
    
    try:
        return float(tod_str)
    except ValueError:
        return 0.0


def message_meets_requirements(msg, qnh_local, temp_local_celsius):
    """
    Applies spatial filters and accurately corrects the barometric altitude 
    using the exact thermodynamic formulas. Includes projected distance.
    """
    try:
        # Check mandatory fields
        campos_obrigatorios = ['x', 'y', 'baroH', 'vx', 'vy', 'trkNr']
        for campo in campos_obrigatorios:
            if not msg.get(campo) or str(msg[campo]).strip() == '':
                return False, f"Empty field '{campo}'", True

        x, y = float(msg['x']), float(msg['y'])
        baroH = float(msg['baroH'])
        vx, vy = float(msg['vx']), float(msg['vy'])

        # 1. Distances calculation
        # Euclidean distance to verify strict spatial boundaries
        distancia_euclidiana_nm = math.sqrt((x - X_THR)**2 + (y - Y_THR)**2) / M_PARA_NM
        
        # Projected distance along the 205º azimuth for precise spatial awareness
        rad_205 = math.radians(205)
        distancia_projetada_nm = ((x - X_THR) * math.sin(rad_205) + (y - Y_THR) * math.cos(rad_205)) / M_PARA_NM

        if distancia_euclidiana_nm > 15:
            return False, "Outside sector", True
            
        # Distance constraint (0 to 11.3 NM from Intermediate Fix to Threshold)
        if not (0 < distancia_euclidiana_nm < 11.3):
            return False, f"Distance = {distancia_euclidiana_nm:.1f} NM", False

        # 2. Bearing constraint (200º to 210º)
        bearing = math.degrees(math.atan2(x - X_THR, y - Y_THR))
        if bearing < 0: bearing += 360
        if not (200 <= bearing <= 210):
            return False, f"Bearing = {bearing:.1f}º", False

        # 3. Exact True Altitude Calculation (Thermodynamic Equations)
        h_p_ft = baroH * 100 
        h_p_m = h_p_ft * FT_PARA_M
        
        expoente_p = -G0 / (LAMBDA_ISA * R_GAS)
        p = P_ISA * (1 + (LAMBDA_ISA * h_p_m) / T_ISA) ** expoente_p
        
        t_local_k = temp_local_celsius + 273.15
        expoente_h = -(LAMBDA_ISA * R_GAS) / G0
        h_true_m = (t_local_k / LAMBDA_ISA) * ((p / qnh_local) ** expoente_h - 1)
        
        h_real = h_true_m / FT_PARA_M
        
        # Altitude constraint (330 ft to 4000 ft)
        if not (330 < h_real < 4000):
            return False, f"Corrected altitude = {h_real:.1f} ft", False

        # 4. Track Angle constraint (20º to 30º)
        chi_g = math.degrees(math.atan2(vx, vy))
        if chi_g < 0: chi_g += 360
        if not (20 < chi_g < 30):
            return False, f"Track = {chi_g:.1f}º", False

        resultado_sucesso = {
            'altitude': h_real,
            'distancia_proj': distancia_projetada_nm
        }
        return True, resultado_sucesso, False

    except ValueError:
        return False, "Numerical error", True


def run_real_time_replay_simulator(caminho_csv, caminho_metar, dia_alvo):
    """
    Main simulator loop that processes the ASTERIX data in real-time order, 
    dynamically updates METAR conditions, and counts validated aircraft.
    """
    print("="*60)
    print("STARTING ARTAS REPLAY SIMULATOR...")
    print("="*60)
    
    # 1. Load Weather Schedule
    cronograma_metar = carregar_cronograma_metar_em_segundos(caminho_metar, dia_alvo)
    tempos_metar = list(cronograma_metar.keys())
    
    # Initial weather state
    qnh_atual, temp_atual = P_ISA, 15.0
    metar_ativo = "Default ISA Conditions"
    indice_proximo_metar = 0

    if tempos_metar:
        metar_ativo = cronograma_metar[tempos_metar[0]]
        qnh_atual, temp_atual = process_metar(metar_ativo)
        indice_proximo_metar = 1 if len(tempos_metar) > 1 else 0
        print(f"[WEATHER] Initial METAR loaded: {metar_ativo}")
        print(f"[WEATHER] QNH: {qnh_atual} hPa | Temp: {temp_atual}ºC\n")

    # Tracking variables
    total_valid_points = 0
    unique_aircraft = set()

    with open(caminho_csv, mode='r', encoding='utf-8') as ficheiro:
        leitor_csv = csv.DictReader(ficheiro, delimiter=';')
        
        try:
            primeira_msg = next(leitor_csv)
        except StopIteration:
            print("The CSV file is empty.")
            return

        ultimo_tempo_bruto = convert_to_seconds(primeira_msg['tod'])
        compensacao_tempo = 0
        t_sim = ultimo_tempo_bruto 
        mensagem_em_espera = primeira_msg

        while mensagem_em_espera is not None:
            t_bruto = convert_to_seconds(mensagem_em_espera['tod'])
            
            # Midnight crossover correction (Adds 24 hours in seconds instead of 1 hour)
            if t_bruto < ultimo_tempo_bruto:
                compensacao_tempo += 86400
                
            ultimo_tempo_bruto = t_bruto
            t_track = t_bruto + compensacao_tempo

            # Wait for the simulation clock to reach the track timestamp
            if t_sim >= t_track:
                
                # Dynamic METAR update logic
                # We check if the time of day (t_track % 86400) has reached the next METAR
                tempo_do_dia = t_track % 86400
                if indice_proximo_metar < len(tempos_metar):
                    proximo_tempo = tempos_metar[indice_proximo_metar]
                    if tempo_do_dia >= proximo_tempo:
                        metar_ativo = cronograma_metar[proximo_tempo]
                        qnh_atual, temp_atual = process_metar(metar_ativo)
                        print(f"\n[CLOCK UPDATE] Virtual Time {t_sim:.0f}s -> New METAR applied:")
                        print(f" -> {metar_ativo}")
                        print(f" -> QNH: {qnh_atual} hPa | Temp: {temp_atual}ºC\n")
                        indice_proximo_metar += 1

                # Validate spatial bounds based on thesis definitions
                valida, resultado, silenciar = message_meets_requirements(
                    mensagem_em_espera, qnh_atual, temp_atual
                )
                
                if valida:
                    trk_nr = mensagem_em_espera['trkNr']
                    unique_aircraft.add(trk_nr)
                    total_valid_points += 1
                    
                    alt_ft = resultado['altitude']
                    dist_nm = resultado['distancia_proj']
                    
                    print(f"[{t_sim:.1f}s] Flight {trk_nr} -> VALIDATED | Dist: {dist_nm:.2f} NM | Alt: {alt_ft:.0f} ft")
                elif not silenciar:
                    # Print rejection reason to monitor the filtering boundary
                    print(f"[{t_sim:.1f}s] Flight {mensagem_em_espera['trkNr']} -> Rejected: {resultado}")

                # Load next message
                try:
                    mensagem_em_espera = next(leitor_csv)
                except StopIteration:
                    mensagem_em_espera = None
            else:
                # Advance simulation clock by 1 second
                t_sim += 1

    # Final Summary Output
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print("Simulation successfully completed.")
    print(f"Total verified aircraft on final approach: {len(unique_aircraft)}")
    print(f"Total valid data points transmitted: {total_valid_points}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Ajusta os caminhos dos ficheiros para os teus dados locais
    nome_do_ficheiro_csv = "dados_1julho_12h16h.csv" 
    nome_do_ficheiro_metar = "METAR_julho.txt"
    dia_alvo = "01"
    
    run_real_time_replay_simulator(nome_do_ficheiro_csv, nome_do_ficheiro_metar, dia_alvo)