import csv
import math
import re
import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# THESIS CONSTANTS
# Constantes físicas e operacionais da aproximação final à pista 02
# ==========================================
X_THR = -433.76  
Y_THR = 277.06   
M_PARA_NM = 1852.0  
FT_PARA_M = 0.3048
P_ISA = 1013.25      
T_ISA = 288.15       
LAMBDA_ISA = -0.0065 
R_GAS = 287.05287    
G0 = 9.80665
TRUE_BEARING_RWY02 = 22.72 # Rumo verdadeiro da pista 02 (graus)

# ==========================================
# MODELO 1: FILTRO DE KALMAN
# Estima o estado do vento minimizando o erro médio quadrático
# ==========================================
class FiltroKalman:
    """
    Filtro de Kalman Linear 1D para o Perfil Vertical de Vento.
    Inclui inicialização com vento de superfície e atualização dinâmica aos 330 ft.
    """
    def __init__(self, vento_inicial_kt, R0_val=34.0, P0_val=1156.0, Kw=0.5, Q_val=0.05):
        # 37 bins de altitude, desde a elevação da pista (330 ft) até ao Intermediate Fix (4000 ft)
        self.bins = np.linspace(330, 4000, 37)
        self.n = len(self.bins)
        self.x = np.full((self.n, 1), vento_inicial_kt)
        self.P = np.eye(self.n) * P0_val
        self.Q = np.eye(self.n) * Q_val 
        self.R0 = R0_val 
        self.Kw = Kw
        
    def predict(self):
        # Passo de previsão: a incerteza (P) aumenta com o tempo (Q) na ausência de medições
        self.P = self.P + self.Q

    def estimar_vento_altitude(self, alt_true):
        # Interpolação linear para estimar o vento numa altitude específica
        idx = np.searchsorted(self.bins, alt_true) - 1
        idx_inf = max(0, min(idx, self.n - 2))
        idx_sup = idx_inf + 1
        h_inf, h_sup = self.bins[idx_inf], self.bins[idx_sup]
        w_inf = max(0.0, min(1.0, (h_sup - alt_true) / (h_sup - h_inf)))
        w_sup = max(0.0, min(1.0, (alt_true - h_inf) / (h_sup - h_inf)))
        return (w_inf * self.x[idx_inf, 0]) + (w_sup * self.x[idx_sup, 0])

    def update(self, observacoes):
        # Atualiza o estado do filtro com as novas observações recebidas das aeronaves
        m = len(observacoes)
        if m == 0: return
        H = np.zeros((m, self.n))
        R = np.zeros((m, m))
        y = np.zeros((m, 1))

        for i, (alt_true, vento_kt, dist_nm) in enumerate(observacoes):
            y[i, 0] = vento_kt
            idx = np.searchsorted(self.bins, alt_true) - 1
            idx_inf = max(0, min(idx, self.n - 2))
            idx_sup = idx_inf + 1
            h_inf, h_sup = self.bins[idx_inf], self.bins[idx_sup]
            
            # Distribui o peso da observação pelas duas bins de altitude mais próximas
            w_inf = max(0.0, min(1.0, (h_sup - alt_true) / (h_sup - h_inf)))
            w_sup = max(0.0, min(1.0, (alt_true - h_inf) / (h_sup - h_inf)))
            H[i, idx_inf], H[i, idx_sup] = w_inf, w_sup
            
            # A variância do ruído de medição aumenta com a distância horizontal
            R[i, i] = self.R0 + (self.Kw * dist_nm)

        # Cálculo do Ganho de Kalman e atualização do estado
        S = np.dot(H, np.dot(self.P, H.T)) + R
        try: K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        except np.linalg.LinAlgError: return
        e = y - np.dot(H, self.x)
        self.x = self.x + np.dot(K, e)
        self.P = np.dot((np.eye(self.n) - np.dot(K, H)), self.P)
        
    def update_vento_superficie(self, vento_kt):
        """Atualiza o valor do vento exclusivamente na altitude de 330 pés (índice 0)"""
        self.x[0, 0] = vento_kt

# ==========================================
# MODELO 2: BASELINE (Zero-Order Hold)
# ==========================================
class ModeloBaseline:
    def __init__(self, vento_inicial_kt):
        self.bins = np.linspace(330, 4000, 37)
        self.n = len(self.bins)
        self.x = np.full((self.n, 1), vento_inicial_kt)
        
    def estimar_vento_altitude(self, alt_true):
        idx = np.searchsorted(self.bins, alt_true) - 1
        idx_inf = max(0, min(idx, self.n - 2))
        idx_sup = idx_inf + 1
        h_inf, h_sup = self.bins[idx_inf], self.bins[idx_sup]
        w_inf = max(0.0, min(1.0, (h_sup - alt_true) / (h_sup - h_inf)))
        w_sup = max(0.0, min(1.0, (alt_true - h_inf) / (h_sup - h_inf)))
        return (w_inf * self.x[idx_inf, 0]) + (w_sup * self.x[idx_sup, 0])

    def update(self, observacoes):
        for alt_true, vento_kt, _ in observacoes:
            idx = (np.abs(self.bins - alt_true)).argmin()
            self.x[idx, 0] = vento_kt
            
    def update_vento_superficie(self, vento_kt):
        """Atualiza o valor do vento exclusivamente na altitude de 330 pés (índice 0)"""
        self.x[0, 0] = vento_kt

# ==========================================
# PRE-PROCESSING FUNCTIONS
# ==========================================
def carregar_cronograma_metar_em_segundos(caminho_metar, dia_alvo):
    # Procura no ficheiro de texto as linhas que correspondem ao dia_alvo
    cronograma = {}
    if os.path.exists(caminho_metar):
        with open(caminho_metar, 'r', encoding='utf-8') as f:
            for linha in f:
                match = re.search(rf'\b{dia_alvo}(\d{{2}})(\d{{2}})Z\b', linha)
                if match:
                    cronograma[(int(match.group(1)) * 3600) + (int(match.group(2)) * 60)] = linha.strip()
    return dict(sorted(cronograma.items()))

def process_metar(metar_str):
    # Extrai QNH, temperatura e calcula a componente de vento de proa (headwind)
    match_qnh = re.search(r'Q(\d{4})', metar_str)
    qnh = float(match_qnh.group(1)) if match_qnh else P_ISA
    
    match_temp = re.search(r'\b(M?\d{2})/(M?\d{2})\b', metar_str)
    temp_celsius = float(match_temp.group(1).replace('M', '-')) if match_temp else 15.0
    
    match_wind = re.search(r'\b(\d{3}|VRB)(\d{2,3})(?:G\d{2,3})?KT\b', metar_str)
    headwind = 0.0
    if match_wind:
        dir_str = match_wind.group(1)
        speed = float(match_wind.group(2))
        if dir_str != 'VRB':
            wind_dir = float(dir_str)
            delta_theta = math.radians(wind_dir - TRUE_BEARING_RWY02)
            headwind = speed * math.cos(delta_theta)
        else:
            headwind = 0.0
            
    return qnh, temp_celsius, headwind

def convert_to_seconds(msg):
    tod_str = msg.get('tod')
    if not tod_str or str(tod_str).strip() == '':
        tod_str = msg.get('rectime', '00:00')
    partes = str(tod_str).strip().split(':')
    if len(partes) == 3: 
        return (float(partes[0]) * 3600) + (float(partes[1]) * 60) + float(partes[2])
    return float(tod_str) if len(partes) != 2 else (float(partes[0]) * 60) + float(partes[1])

def message_meets_requirements(msg, qnh_local, temp_local_celsius):
    # Verifica se a mensagem respeita os limites espaciais da aproximação final (Tese: Capítulo 3.2.1)
    try:
        campos = ['x', 'y', 'baroH', 'vx', 'vy', 'trkNr']
        if any(not msg.get(c) or str(msg[c]).strip() == '' for c in campos): return False, None
        
        x, y = float(msg['x']), float(msg['y'])
        baroH = float(msg['baroH'])
        vx, vy = float(msg['vx']), float(msg['vy'])
        
        dist_nm = math.sqrt((x - X_THR)**2 + (y - Y_THR)**2) / M_PARA_NM
        if not (0 < dist_nm < 11.3): return False, None
            
        bearing = (math.degrees(math.atan2(x - X_THR, y - Y_THR)) + 360) % 360
        if not (197.72 <= bearing <= 207.72): return False, None

        p = P_ISA * (1 + (LAMBDA_ISA * (baroH * 100 * FT_PARA_M)) / T_ISA) ** (-G0 / (LAMBDA_ISA * R_GAS))
        h_real = ((temp_local_celsius + 273.15) / LAMBDA_ISA) * ((p / qnh_local) ** (-(LAMBDA_ISA * R_GAS) / G0) - 1) / FT_PARA_M
        if not (330 < h_real < 4000): return False, None

        chi_g = (math.degrees(math.atan2(vx, vy)) + 360) % 360
        if not (17.72 <= chi_g <= 27.72): return False, None

        dist_proj_nm = ((x - X_THR) * math.sin(math.radians(202.72)) + (y - Y_THR) * math.cos(math.radians(202.72))) / M_PARA_NM
        return True, {'altitude': h_real, 'distancia_proj': dist_proj_nm}
    except ValueError:
        return False, None

def calcular_vento_local(msg, temp_local_celsius):
    # Deriva a observação de vento subtraindo a ground speed à true airspeed (Tese: Capítulo 3.2.2)
    try:
        if not msg.get('iar') or str(msg['iar']).strip() == '': return None
        iar_kt, vx, vy, baroH = float(msg['iar']), float(msg['vx']), float(msg['vy']), float(msg['baroH'])
        vg_kt = math.sqrt(vx**2 + vy**2) * (3600 / M_PARA_NM) 
        p_pa = (P_ISA * 100) * (1 + (LAMBDA_ISA * (baroH * 100 * FT_PARA_M)) / T_ISA) ** (-G0 / (LAMBDA_ISA * R_GAS))
        rho_local = p_pa / (R_GAS * (temp_local_celsius + 273.15))
        return (iar_kt * math.sqrt((P_ISA * 100 / (R_GAS * T_ISA)) / rho_local)) - vg_kt
    except ValueError: return None

def formatar_tempo(segundos_absolutos):
    horas = int(segundos_absolutos // 3600)
    minutos = int((segundos_absolutos % 3600) // 60)
    segundos = int(segundos_absolutos % 60)
    return f"{horas:02d}:{minutos:02d}:{segundos:02d}"

# ==========================================
# MAIN SIMULATION & EVALUATION
# ==========================================
def run_and_visualise(caminho_csv, caminho_metar, dia_alvo, Kw_val, Q_val, R0_val):
    print(f"A iniciar simulação comparativa (Kw = {Kw_val} | Q = {Q_val} | R0 = {R0_val})...")
    
    cronograma_metar = carregar_cronograma_metar_em_segundos(caminho_metar, dia_alvo)
    tempos_metar = list(cronograma_metar.keys())
    
    qnh_atual, temp_atual, vento_atual = process_metar(cronograma_metar[tempos_metar[0]]) if tempos_metar else (P_ISA, 15.0, 0.0)
    indice_proximo_metar = 1 if len(tempos_metar) > 1 else 0

    kf = FiltroKalman(vento_inicial_kt=vento_atual, R0_val=R0_val, Kw=Kw_val, Q_val=Q_val)
    baseline = ModeloBaseline(vento_inicial_kt=vento_atual)
    
    historico_t = []
    observacoes_recentes = [] 
    historico_perfis_kf = []
    historico_perfis_base = []
    
    # ---------------------------------------------------------
    # NOVO: Lista para gravar a evolução do vento à superfície
    # ---------------------------------------------------------
    historico_vento_superficie = []
    
    erros_kf = []
    erros_base = []
    total_pontos = 0

    alvos_horas = [1, 2, 3, 4]
    snapshots_data = {h: {'kf': None, 'base': None, 'obs_alt': [], 'obs_wind': []} for h in alvos_horas}

    plt.ion()
    fig_live, ax_live = plt.subplots(figsize=(10.5, 9))
    fig_live.canvas.manager.set_window_title('Real-Time Model Comparison')
    
    # Gráficos com legendas em Inglês Britânico
    scatter_obs = ax_live.scatter([], [], color='dodgerblue', s=15, alpha=0.5, label='Recent Raw Observations (15 mins)')
    line_kf, = ax_live.plot([], [], color='red', linewidth=2.5, label='Estimated Profile (Kalman)')
    line_base, = ax_live.plot([], [], color='green', linestyle='--', linewidth=2.5, label='Baseline Profile (Latest Value)')
    
    ax_live.axhline(y=4000, color='purple', linestyle='--', alpha=0.5, label='Intermediate Fix (4000 ft)')
    ax_live.axhline(y=330, color='orange', linestyle='--', alpha=0.5, label='Runway Elevation (330 ft)')
    
    ax_live.set_ylim(0, 4500)
    ax_live.set_xlim(-10, 40)
    ax_live.set_title("Real-Time Vertical Headwind Profile Estimation", fontweight='bold')
    ax_live.set_xlabel("Calculated Headwind (kt)")
    ax_live.set_ylabel("True Altitude (ft)")
    ax_live.grid(True, linestyle='--', alpha=0.6)
    
    ax_live.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.subplots_adjust(right=0.75)
    
    caixa_tempo = ax_live.text(0.05, 0.95, '', transform=ax_live.transAxes, fontsize=11,
                              verticalalignment='top', 
                              bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.8))
    plt.show()

    with open(caminho_csv, mode='r', encoding='utf-8') as ficheiro:
        leitor_csv = csv.DictReader(ficheiro, delimiter=';')
        try: mensagem_em_espera = next(leitor_csv)
        except StopIteration: return

        ultimo_tempo_bruto = convert_to_seconds(mensagem_em_espera)
        t_sim = int(ultimo_tempo_bruto)
        t_start_sim = t_sim
        compensacao_tempo = 0

        while mensagem_em_espera is not None:
            observacoes_do_segundo = []
            
            while mensagem_em_espera is not None:
                t_bruto = convert_to_seconds(mensagem_em_espera)
                if t_bruto < ultimo_tempo_bruto - 60: 
                    compensacao_tempo += 3600 
                    
                ultimo_tempo_bruto = t_bruto
                t_track = t_bruto + compensacao_tempo
                
                if t_track <= t_sim:
                    valida, resultado = message_meets_requirements(mensagem_em_espera, qnh_atual, temp_atual)
                    if valida:
                        headwind_kt = calcular_vento_local(mensagem_em_espera, temp_atual)
                        if headwind_kt is not None:
                            alt_ft = resultado['altitude']
                            dist_nm = resultado['distancia_proj']
                            observacoes_do_segundo.append((alt_ft, headwind_kt, dist_nm))
                            observacoes_recentes.append((t_sim, alt_ft, headwind_kt))
                            
                            vento_pred_kf = kf.estimar_vento_altitude(alt_ft)
                            vento_pred_base = baseline.estimar_vento_altitude(alt_ft)
                            total_pontos += 1
                            
                            # ==========================================
                            # IMPLEMENTAÇÃO DO WARM-UP TEMPORAL 
                            # O erro só é calculado 15 minutos (900 seg) 
                            # após o início da simulação.
                            # ==========================================
                            if (t_sim - t_start_sim) >= 900:
                                erros_kf.append(vento_pred_kf - headwind_kt)
                                erros_base.append(vento_pred_base - headwind_kt)

                    try: mensagem_em_espera = next(leitor_csv)
                    except StopIteration: mensagem_em_espera = None
                else: break 

            if observacoes_do_segundo:
                kf.update(observacoes_do_segundo)
                baseline.update(observacoes_do_segundo)
            kf.predict()
            
            for h in alvos_horas:
                if t_sim == t_start_sim + (h * 3600):
                    snapshots_data[h]['kf'] = np.copy(kf.x)
                    snapshots_data[h]['base'] = np.copy(baseline.x)
                    snapshots_data[h]['obs_alt'] = [obs[1] for obs in observacoes_recentes]
                    snapshots_data[h]['obs_wind'] = [obs[2] for obs in observacoes_recentes]
            
            if t_sim % 10 == 0:
                historico_t.append(t_sim)
                historico_perfis_kf.append(np.copy(kf.x))
                historico_perfis_base.append(np.copy(baseline.x))
                
                # ---------------------------------------------------------
                # NOVO: Gravar o valor instantâneo do vento à superfície
                # ---------------------------------------------------------
                historico_vento_superficie.append(vento_atual)
            
            if t_sim % 60 == 0:
                line_kf.set_data(kf.x, kf.bins)
                line_base.set_data(baseline.x, baseline.bins)
                
                observacoes_recentes = [obs for obs in observacoes_recentes if (t_sim - obs[0]) <= 900]
                if observacoes_recentes:
                    _, alts_recentes, ventos_recentes = zip(*observacoes_recentes)
                    scatter_obs.set_offsets(np.c_[ventos_recentes, alts_recentes])
                    min_wind, max_wind = min(ventos_recentes), max(ventos_recentes)
                    if min_wind < ax_live.get_xlim()[0] or max_wind > ax_live.get_xlim()[1]:
                        ax_live.set_xlim(min_wind - 5, max_wind + 5)
                
                caixa_tempo.set_text(f"Virtual Time: {formatar_tempo(t_sim)}\nMETAR Temp: {temp_atual}ºC\nSfc Wind: {vento_atual:.1f} kt")
                plt.pause(0.01)

            tempo_do_dia = t_sim % 86400
            if indice_proximo_metar < len(tempos_metar) and tempo_do_dia >= tempos_metar[indice_proximo_metar]:
                qnh_atual, temp_atual, vento_atual = process_metar(cronograma_metar[tempos_metar[indice_proximo_metar]])
                kf.update_vento_superficie(vento_atual)
                baseline.update_vento_superficie(vento_atual)
                indice_proximo_metar += 1

            t_sim += 1

    plt.ioff()
    plt.close(fig_live)

    # ==========================================
    # CÁLCULO ESTATÍSTICO E PAINEL FINAL
    # ==========================================
    erros_kf_np = np.array(erros_kf)
    erros_base_np = np.array(erros_base)
    
    if len(erros_kf_np) > 0:
        rmse_kf, mae_kf = np.sqrt(np.mean(erros_kf_np**2)), np.mean(np.abs(erros_kf_np))
        rmse_base, mae_base = np.sqrt(np.mean(erros_base_np**2)), np.mean(np.abs(erros_base_np))
    else:
        rmse_kf = mae_kf = rmse_base = mae_base = 0.0

    matriz_historico_kf = np.array(historico_perfis_kf)
    matriz_historico_base = np.array(historico_perfis_base)

    media_coluna_temporal_kf = np.mean(matriz_historico_kf, axis=1).flatten()
    media_coluna_temporal_base = np.mean(matriz_historico_base, axis=1).flatten()
    
    horas_relativas = [(t - t_start_sim) / 3600 for t in historico_t]

    print("\n" + "="*60)
    print("MODEL COMPARISON EVALUATION")
    print("="*60)
    print(f"Total Valid Observations Found: {total_pontos}")
    print(f"Total Valid Observations Evaluated (Post Warm-up): {len(erros_kf_np)}")
    print("------------------------------------------------------------")
    print("KALMAN FILTER:")
    print(f"  RMSE : {rmse_kf:.2f} kt")
    print(f"  MAE  : {mae_kf:.2f} kt")
    print("------------------------------------------------------------")
    print("BASELINE MODEL:")
    print(f"  RMSE : {rmse_base:.2f} kt")
    print(f"  MAE  : {mae_base:.2f} kt")
    print("="*60 + "\n")

    for h in alvos_horas:
        if snapshots_data[h]['kf'] is not None:
            fig_snap, ax_snap = plt.subplots(figsize=(10, 8))
            fig_snap.canvas.manager.set_window_title(f'Snapshot at T = {h} Hour(s)')
            
            ax_snap.plot(snapshots_data[h]['kf'], kf.bins, color='red', linewidth=2.5, label='Kalman Filter Profile')
            ax_snap.plot(snapshots_data[h]['base'], kf.bins, color='green', linestyle='--', linewidth=2.5, label='Baseline Profile')
            
            if snapshots_data[h]['obs_wind']:
                ax_snap.scatter(snapshots_data[h]['obs_wind'], snapshots_data[h]['obs_alt'], color='dodgerblue', s=20, alpha=0.6, label='Raw Observations (Last 15 mins)')
            
            ax_snap.axhline(y=4000, color='purple', linestyle='--', alpha=0.6, label='Intermediate Fix (4000 ft)')
            ax_snap.axhline(y=330, color='orange', linestyle='--', alpha=0.6, label='Runway Elevation (330 ft)')
            
            ax_snap.set_ylim(0, 4500)
            ax_snap.set_title(f"Vertical Headwind Profile Snapshot (T = {h}h)", fontweight='bold')
            ax_snap.set_xlabel("Calculated Headwind (kt)")
            ax_snap.set_ylabel("True Altitude (ft)")
            ax_snap.grid(True, linestyle='--', alpha=0.6)
            
            ax_snap.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
            fig_snap.subplots_adjust(right=0.7)
            
            plt.show(block=False)

    # ---------------------------------------------------------
    # PRIMEIRA JANELA FINAL: EVOLUÇÃO TEMPORAL
    # ---------------------------------------------------------
    fig_time, ax_time = plt.subplots(figsize=(11, 7))
    fig_time.canvas.manager.set_window_title('Temporal Evolution of the Column Mean Headwind')

    ax_time.plot(horas_relativas, media_coluna_temporal_kf, color='red', linewidth=2, label='Kalman Filter Column Mean')
    ax_time.plot(horas_relativas, media_coluna_temporal_base, color='green', linestyle='--', linewidth=1.5, alpha=0.8, label='Baseline Column Mean')
    
    # ---------------------------------------------------------
    # NOVO: Desenhar a terceira linha referente ao vento de superfície
    # ---------------------------------------------------------
    ax_time.plot(horas_relativas, historico_vento_superficie, color='blue', linestyle='-.', linewidth=1.5, alpha=0.8, label='Surface Headwind')
    
    ax_time.set_title("Temporal Evolution of the Column Mean Headwind", fontweight='bold')
    ax_time.set_xlabel("Elapsed Time (hours)")
    ax_time.set_ylabel("Mean Column Headwind (kt)")
    ax_time.grid(True, linestyle='--', alpha=0.6)
    
    ax_time.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    fig_time.subplots_adjust(right=0.7)

    # ---------------------------------------------------------
    # SEGUNDA JANELA FINAL: DISTRIBUIÇÃO DE ERROS (BOXPLOTS)
    # ---------------------------------------------------------
    fig_box, ax_box = plt.subplots(figsize=(9, 7))
    fig_box.canvas.manager.set_window_title('Residual Error Distribution Comparison')

    box_data = [erros_kf_np, erros_base_np]
    bp = ax_box.boxplot(box_data, vert=True, patch_artist=True, labels=['Kalman Filter', 'Baseline Model'])
    
    colors = ['lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
    for median in bp['medians']:
        median.set(color='black', linewidth=1.5)

    ax_box.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Zero Error Reference')
    ax_box.set_title("Residual Error Distribution Comparison", fontweight='bold')
    ax_box.set_ylabel("Estimation Error (kt)")
    
    ax_box.text(1, ax_box.get_ylim()[0] * 0.95, f"RMSE: {rmse_kf:.2f}\nMAE: {mae_kf:.2f}", ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.6))
    ax_box.text(2, ax_box.get_ylim()[0] * 0.95, f"RMSE: {rmse_base:.2f}\nMAE: {mae_base:.2f}", ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.6))
    
    ax_box.grid(True, linestyle='--', alpha=0.6)
    
    ax_box.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    fig_box.subplots_adjust(right=0.7)

    fig_time.tight_layout()
    fig_box.tight_layout()
    plt.show(block=True)

if __name__ == "__main__":
    # NOVOS FICHEIROS DE ENTRADA E PARÂMETROS CONFIGURADOS AQUI
    nome_do_ficheiro_csv = "dados_16março_12h16h.csv" 
    
    # Assumi que o teu ficheiro METAR se chama "METAR_maio.txt" para acompanhar a lógica do teu projeto.
    # Caso se chame diferente, ajusta apenas esta linha.
    nome_do_ficheiro_metar = "METAR_março.txt" 
    
    dia_alvo = "16" # Procura o padrão "16" nos dias do relatório METAR de Março
    
    # AFINAÇÃO DOS PARÂMETROS
    Kw_teste = 1 
    R0_teste = 34.0  
    
    # O valor de Q foi regressado a 0.03 porque dia 6 de maio caracteriza-se por "Low Headwind" (3.9 kt de média).
    Q_teste = 0.03
    
    run_and_visualise(nome_do_ficheiro_csv, nome_do_ficheiro_metar, dia_alvo, Kw_teste, Q_teste, R0_teste)