import csv
import math
import re
import os
import numpy as np

# ==========================================
# THESIS CONSTANTS
# ==========================================
X_THR = -433.76  
Y_THR = 277.06   
M_PARA_NM = 1852.0  
FT_PARA_M = 0.3048

# ==========================================
# STANDARD ATMOSPHERE (ISA) CONSTANTS
# ==========================================
P_ISA = 1013.25      
T_ISA = 288.15       
LAMBDA_ISA = -0.0065 
R_GAS = 287.05287    
G0 = 9.80665         


class FiltroKalman:
    """
    Implementação do Filtro de Kalman Linear 1D para o Perfil de Vento.
    O vetor de estado contém 37 níveis de altitude.
    """
    def __init__(self, R0=34.0, P0_val=1156.0, Kw=0.5):
        # 37 níveis de altitude desde os 330 ft até aos 4000 ft
        self.bins = np.linspace(330, 4000, 37)
        self.n = len(self.bins)
        
        # Vetor de Estado (Perfil do vento estimado em kt)
        self.x = np.zeros((self.n, 1))
        
        # Matriz de Covariância do Erro (P) e Ruído de Processo (Q)
        self.P = np.eye(self.n) * P0_val
        self.Q = np.eye(self.n) * 1.0  # Matriz Identidade
        
        # Parâmetros de medição
        self.R0 = R0
        self.Kw = Kw
        
    def predict(self):
        """
        Passo de Previsão Temporal. 
        A incerteza P cresce com a passagem do tempo (adição de Q).
        O vetor x não se altera porque a Matriz de Transição M é a Identidade.
        """
        self.P = self.P + self.Q

    def update(self, observacoes):
        """
        Passo de Análise (Matricial em Bloco).
        observacoes é uma lista de tuplos: (altitude_true_ft, vento_medido_kt, distancia_ao_glide_nm)
        """
        m = len(observacoes)
        if m == 0:
            return

        H = np.zeros((m, self.n))
        R = np.zeros((m, m))
        y = np.zeros((m, 1))

        for i, (alt_true, vento_kt, dist_nm) in enumerate(observacoes):
            # 1. Popular o vetor de medição
            y[i, 0] = vento_kt
            
            # 2. Calcular os pesos (Interpolação Linear)
            # Encontrar o índice do bin inferior mais próximo
            idx = np.searchsorted(self.bins, alt_true) - 1
            
            # Garantir limites
            if idx < 0:
                idx_inf, idx_sup = 0, 1
            elif idx >= self.n - 1:
                idx_inf, idx_sup = self.n - 2, self.n - 1
            else:
                idx_inf, idx_sup = idx, idx + 1
                
            h_inf = self.bins[idx_inf]
            h_sup = self.bins[idx_sup]
            
            W_inf = (h_sup - alt_true) / (h_sup - h_inf)
            W_sup = (alt_true - h_inf) / (h_sup - h_inf)
            
            # Prevenir extrapolações anómalas
            W_inf = max(0.0, min(1.0, W_inf))
            W_sup = max(0.0, min(1.0, W_sup))
            
            H[i, idx_inf] = W_inf
            H[i, idx_sup] = W_sup
            
            # 3. Popular a Matriz de Ruído da Medição R dependente da distância
            R[i, i] = self.R0 + (self.Kw * dist_nm)

        # 4. Cálculo do Ganho de Kalman (K) e Atualização do Estado
        # S = H * P * H^T + R
        S = np.dot(H, np.dot(self.P, H.T)) + R
        
        # K = P * H^T * inv(S)
        try:
            K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        except np.linalg.LinAlgError:
            print("[WARNING] Singularidade matricial na inversão. Salto de atualização.")
            return
            
        # Inovação (e)
        e = y - np.dot(H, self.x)
        
        # Atualizar Estado x = x + K*e
        self.x = self.x + np.dot(K, e)
        
        # Atualizar Covariância P = (I - K*H) * P
        I = np.eye(self.n)
        self.P = np.dot((I - np.dot(K, H)), self.P)


def carregar_cronograma_metar_em_segundos(caminho_metar, dia_alvo):
    cronograma = {}
    if not os.path.exists(caminho_metar):
        print(f"[WARNING] File {caminho_metar} not found. Default ISA weather will be used.")
        return cronograma
        
    with open(caminho_metar, 'r', encoding='utf-8') as f:
        for linha in f:
            padrao = rf'\b{dia_alvo}(\d{{2}})(\d{{2}})Z\b'
            match = re.search(padrao, linha)
            if match:
                hora = int(match.group(1))
                minuto = int(match.group(2))
                tempo_em_segundos = (hora * 3600) + (minuto * 60)
                cronograma[tempo_em_segundos] = linha.strip()
                
    return dict(sorted(cronograma.items()))

def process_metar(metar_str):
    match_qnh = re.search(r'Q(\d{4})', metar_str)
    qnh = float(match_qnh.group(1)) if match_qnh else P_ISA
    
    match_temp = re.search(r'\b(M?\d{2})/(M?\d{2})\b', metar_str)
    if match_temp:
        temp_str = match_temp.group(1)
        temp_celsius = float(temp_str.replace('M', '-'))
    else:
        temp_celsius = 15.0 

    return qnh, temp_celsius

def convert_to_seconds(tod_str):
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
    Verifica se a mensagem cumpre os requisitos espaciais e direcionais
    para o glide path da pista 02.
    """
    try:
        campos_obrigatorios = ['x', 'y', 'baroH', 'vx', 'vy', 'trkNr']
        for campo in campos_obrigatorios:
            if not msg.get(campo) or str(msg[campo]).strip() == '':
                return False, f"Empty field '{campo}'", True

        x, y = float(msg['x']), float(msg['y'])
        baroH = float(msg['baroH'])
        vx, vy = float(msg['vx']), float(msg['vy'])

        # Distância Euclidiana 2D
        distancia_euclidiana_nm = math.sqrt((x - X_THR)**2 + (y - Y_THR)**2) / M_PARA_NM
        
        # 1. Distância projetada com base no rumo de 202.72º (22.72º + 180º)
        rad_202_72 = math.radians(202.72)
        distancia_projetada_nm = ((x - X_THR) * math.sin(rad_202_72) + (y - Y_THR) * math.cos(rad_202_72)) / M_PARA_NM

        if distancia_euclidiana_nm > 15:
            return False, "Outside sector", True
        if not (0 < distancia_euclidiana_nm < 11.3):
            return False, f"Distance = {distancia_euclidiana_nm:.1f} NM", False

        # 2. Rumo relativo à cabeceira (Bearing)
        bearing = math.degrees(math.atan2(x - X_THR, y - Y_THR))
        if bearing < 0: bearing += 360
        # O novo cone de aceitação de bearing é 202.72º ± 5º (197.72 a 207.72)
        if not (197.72 <= bearing <= 207.72):
            return False, f"Bearing = {bearing:.1f}º", False

        h_p_ft = baroH * 100 
        h_p_m = h_p_ft * FT_PARA_M
        expoente_p = -G0 / (LAMBDA_ISA * R_GAS)
        p = P_ISA * (1 + (LAMBDA_ISA * h_p_m) / T_ISA) ** expoente_p
        
        t_local_k = temp_local_celsius + 273.15
        expoente_h = -(LAMBDA_ISA * R_GAS) / G0
        h_true_m = (t_local_k / LAMBDA_ISA) * ((p / qnh_local) ** expoente_h - 1)
        h_real = h_true_m / FT_PARA_M
        
        if not (330 < h_real < 4000):
            return False, f"Corrected altitude = {h_real:.1f} ft", False

        # 3. Track Angle da Aeronave (Chi_g)
        chi_g = math.degrees(math.atan2(vx, vy))
        if chi_g < 0: chi_g += 360
        # O novo limite direcional de voo é 22.72º ± 5º (17.72 a 27.72)
        if not (17.72 <= chi_g <= 27.72):
            return False, f"Track = {chi_g:.1f}º", False

        resultado_sucesso = {
            'altitude': h_real,
            'distancia_proj': distancia_projetada_nm
        }
        return True, resultado_sucesso, False
    except ValueError:
        return False, "Numerical error", True

def calcular_vento_local(msg, temp_local_celsius):
    try:
        if not msg.get('iar') or str(msg['iar']).strip() == '':
            return None 

        iar_kt = float(msg['iar'])
        vx = float(msg['vx'])
        vy = float(msg['vy'])
        baroH = float(msg['baroH'])

        vg_ms = math.sqrt(vx**2 + vy**2)
        vg_kt = vg_ms * (3600 / M_PARA_NM) 

        p0_pa = P_ISA * 100 
        hp_m = baroH * 100 * FT_PARA_M 
        expoente_p = -G0 / (LAMBDA_ISA * R_GAS)
        p_pa = p0_pa * (1 + (LAMBDA_ISA * hp_m) / T_ISA) ** expoente_p

        t_local_k = temp_local_celsius + 273.15
        rho_local = p_pa / (R_GAS * t_local_k)
        rho_0 = p0_pa / (R_GAS * T_ISA)

        vtas_kt = iar_kt * math.sqrt(rho_0 / rho_local)
        headwind_kt = vtas_kt - vg_kt
        return headwind_kt
    except ValueError:
        return None


def run_real_time_replay_simulator(caminho_csv, caminho_metar, dia_alvo):
    print("="*60)
    print("STARTING ARTAS REPLAY SIMULATOR WITH KALMAN FILTER...")
    print("="*60)
    
    cronograma_metar = carregar_cronograma_metar_em_segundos(caminho_metar, dia_alvo)
    tempos_metar = list(cronograma_metar.keys())
    
    qnh_atual, temp_atual = P_ISA, 15.0
    metar_ativo = "Default ISA Conditions"
    indice_proximo_metar = 0

    if tempos_metar:
        metar_ativo = cronograma_metar[tempos_metar[0]]
        qnh_atual, temp_atual = process_metar(metar_ativo)
        indice_proximo_metar = 1 if len(tempos_metar) > 1 else 0
        print(f"[WEATHER] Initial METAR loaded: {metar_ativo}")

    # Inicialização do Filtro de Kalman
    kf = FiltroKalman(R0=34.0, P0_val=1156.0, Kw=0.5)

    total_valid_points = 0
    unique_aircraft = set()

    with open(caminho_csv, mode='r', encoding='utf-8') as ficheiro:
        leitor_csv = csv.DictReader(ficheiro, delimiter=';')
        
        try:
            mensagem_em_espera = next(leitor_csv)
        except StopIteration:
            print("The CSV file is empty.")
            return

        ultimo_tempo_bruto = convert_to_seconds(mensagem_em_espera['tod'])
        compensacao_tempo = 0
        t_sim = ultimo_tempo_bruto 

        # Ciclo principal do Relógio Virtual
        while mensagem_em_espera is not None:
            
            observacoes_do_segundo = []
            
            while mensagem_em_espera is not None:
                t_bruto = convert_to_seconds(mensagem_em_espera['tod'])
                
                if t_bruto < ultimo_tempo_bruto:
                    compensacao_tempo += 86400
                    
                ultimo_tempo_bruto = t_bruto
                t_track = t_bruto + compensacao_tempo
                
                if t_track <= t_sim:
                    valida, resultado, silenciar = message_meets_requirements(mensagem_em_espera, qnh_atual, temp_atual)
                    
                    if valida:
                        trk_nr = mensagem_em_espera['trkNr']
                        unique_aircraft.add(trk_nr)
                        total_valid_points += 1
                        
                        headwind_kt = calcular_vento_local(mensagem_em_espera, temp_atual)
                        if headwind_kt is not None:
                            alt_ft = resultado['altitude']
                            dist_nm = resultado['distancia_proj']
                            observacoes_do_segundo.append((alt_ft, headwind_kt, dist_nm))
                            print(f"[{t_sim:.0f}s] MSG Valida -> Voo {trk_nr} | Vento: {headwind_kt:.1f} kt | Dist: {dist_nm:.1f} NM")

                    try:
                        mensagem_em_espera = next(leitor_csv)
                    except StopIteration:
                        mensagem_em_espera = None
                else:
                    break 

            if observacoes_do_segundo:
                kf.update(observacoes_do_segundo)
                print(f"[{t_sim:.0f}s] *** Filtro Atualizado (Análise) para {len(observacoes_do_segundo)} observação(ões) ***")
            
            kf.predict()
            
            tempo_do_dia = t_sim % 86400
            if indice_proximo_metar < len(tempos_metar):
                proximo_tempo = tempos_metar[indice_proximo_metar]
                if tempo_do_dia >= proximo_tempo:
                    metar_ativo = cronograma_metar[proximo_tempo]
                    qnh_atual, temp_atual = process_metar(metar_ativo)
                    print(f"\n[CLOCK UPDATE] Virtual Time {t_sim:.0f}s -> New METAR applied: {metar_ativo}\n")
                    indice_proximo_metar += 1

            t_sim += 1

    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"Total verified aircraft on final approach: {len(unique_aircraft)}")
    print(f"Total valid wind observations fed to filter: {total_valid_points}")
    print("O Vetor Final de Estado Estimado (Perfil do Vento) encontra-se em 'kf.x'")
    print("="*60 + "\n")


if __name__ == "__main__":
    nome_do_ficheiro_csv = "dados_1julho_12h16h.csv" 
    nome_do_ficheiro_metar = "METAR_julho.txt"
    dia_alvo = "01"
    
    run_real_time_replay_simulator(nome_do_ficheiro_csv, nome_do_ficheiro_metar, dia_alvo)