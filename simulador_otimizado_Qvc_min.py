import numpy as np
from scipy.optimize import fsolve, minimize, Bounds, NonlinearConstraint, BFGS
import matplotlib.pyplot as plt
# Parâmetros do sistema
area_deposito = 185  # m²
altura_inicial = 4  # m
altura_max = 7  # m
altura_min = 2  # m
tempo_total = 24  # horas
dist_bomba_dep = 150  # m
nDutyCycles = 2
h0 = altura_inicial

tarifa_energia = np.array([
    0.0713, 0.0713, 0.0651, 0.0651, 0.0593, 0.0593, 0.0778, 0.0778,
    0.0851, 0.0851, 0.0923, 0.0923, 0.0968, 0.0968, 0.10094, 0.10094,
    0.10132, 0.10132, 0.10230, 0.10230, 0.10189, 0.10189, 0.10132, 0.10132
])

# Constante para cálculo de perdas de carga
coef_perdas = (32 * 0.02) / (0.3**5 * 9.81 * np.pi**2)

# Funções dos caudais
def calcular_Q_r(t):
    return (-0.004 * t**3 + 0.09 * t**2 + 0.1335 * t + 20)/3600

def calcular_Q_vc_max(t):  ##na verdade é Q_vc_min, mas como foi adaptado do simulador original para Q_vc_max, mantive o nome para não complicar e só troquei a função
    return (1.19333e-7 * t**7 - 6.54846e-5 * t**6 + 4.1432e-3 * t**5
    - 0.100585 * t**4 + 1.05575 * t**3 - 3.85966 * t**2 - 1.32657 * t + 75.393)/3600

def calcular_Q_pump(Q_r, altura_atual, dist_bomba_dep):
    def equation(Qp):
        h_pump = 260 - 0.002 * Qp**2 * 3600**2
        perdas_totais = coef_perdas*5000*(Qp - Q_r)**2 + coef_perdas*2500*Q_r**2
        h_required = altura_atual + dist_bomba_dep + perdas_totais
        return h_pump - h_required

    Qp_initial_guess = 1
    Qp_solution = fsolve(equation, Qp_initial_guess)[0]
    return Qp_solution

def calcular_custo(Q_pump, preço_por_hora):
    eficiencia = 0.65
    h_pump = 260 - 0.002 * Q_pump**2 * 3600**2
    densidade = 1000
    g = 9.81
    custo_bomba = (Q_pump * densidade * g * h_pump)/(eficiencia * 1000) * preço_por_hora
    return custo_bomba

# Função simulador principal
def simulador_hidraulico(x):
    estado_bomba = np.zeros(24)
    for i in range(nDutyCycles):
        inicio = x[i]-1     
        duracao = x[i + nDutyCycles]
        for t in range(24):
            if inicio <= t < inicio + duracao:
                estado_bomba[t] = 1

    resultados_Q_r = []
    resultados_Q_vc_max = []
    resultados_Q_pump = []
    resultados_h_atual = [h0]
    custo_total = 0
    custo_por_hora = []          # Nova lista para custos horários
    energia_por_hora = []         # Nova lista para energia horária

    for t in range(24):
        Q_r = calcular_Q_r(t)
        Q_vc_max = calcular_Q_vc_max(t)
        if estado_bomba[t] == 1:
            Q_pump = calcular_Q_pump(Q_r, resultados_h_atual[t], dist_bomba_dep)
            # Cálculo da energia gasta nesta hora (em kW)
            eficiencia = 0.65
            h_pump = 260 - 0.002 * Q_pump**2 * 3600**2
            densidade = 1000
            g = 9.81
            energia = (Q_pump * densidade * g * h_pump) / (eficiencia * 1000) #kWh
            energia_por_hora.append(energia)
            # Cálculo do custo nesta hora
            custo_hora = calcular_custo(Q_pump, tarifa_energia[t])
            custo_por_hora.append(custo_hora)
            custo_total += custo_hora
        else:
            Q_pump = 0
            energia_por_hora.append(0)      # Bomba desligada = energia 0
            custo_por_hora.append(0)       # Bomba desligada = custo 0

        delta_h = ((Q_pump - Q_r - Q_vc_max) / area_deposito) * 3600
        nova_altura = resultados_h_atual[t] + delta_h
        resultados_Q_r.append(Q_r)
        resultados_Q_vc_max.append(Q_vc_max)
        resultados_Q_pump.append(Q_pump)
        resultados_h_atual.append(nova_altura)

    return {
        'fObj': custo_total,
        'g1': resultados_h_atual[1:],  
        'Q_r': resultados_Q_r,
        'Q_vc_max': resultados_Q_vc_max,
        'Q_pump': resultados_Q_pump,
        'h': resultados_h_atual[1:],
        'estado_bomba': estado_bomba,
        'custo_por_hora': custo_por_hora,   # Novo campo
        'energia_por_hora': energia_por_hora # Novo campo
    }

# Restrições
def fun_constr_1(x):
    res = simulador_hidraulico(x)
    return res['g1']  # altura a cada hora

def fun_constr_2(x):
    constr2 = []
    for i in range(nDutyCycles - 1):
        constr2.append(x[i] + x[i + nDutyCycles] - x[i + 1])
    constr2.append(x[nDutyCycles - 1] + x[2 * nDutyCycles - 1] - 24)
    return constr2

def fun_constr_3(x):
    res = simulador_hidraulico(x)
    return h0 - res['g1'][-1]  # garantir que altura final >= inicial

def fun_obj(x):
    res = simulador_hidraulico(x)
    return res['fObj']

# Otimização
FD_dx = 1e-4
bounds = Bounds([0.01] * (2 * nDutyCycles), [23.99] * (2 * nDutyCycles))
c1 = NonlinearConstraint(fun_constr_1, altura_min, altura_max, jac='2-point', hess=BFGS())
c2 = NonlinearConstraint(fun_constr_2, -np.inf, 0.0, jac='2-point', hess=BFGS())
c3 = NonlinearConstraint(fun_constr_3, -np.inf, 0.0, jac='2-point', hess=BFGS())

x0 = [3, 14] + [7, 5]  # palpite inicial

res_optim = minimize(
    fun_obj, x0,
    method='trust-constr',
    jac='2-point',
    constraints=[c1, c2, c3],
    bounds=bounds,
    options={
        'verbose': 3,
        'maxiter': 100,
        'gtol': 1e-2,          
        'xtol': 1e-6,
        'barrier_tol': 1e-8
    }
)

# Resultados finais
final_sim = simulador_hidraulico(res_optim.x)
print("\nResultados completos:")
print("Hora | Bomba | Q_r (m³/s) | Q_vc_min (m³/s) | Q_pump (m³/s) | altura_atual (m)")
print("-----|-------|------------|-----------------|---------------|------------------")

for t in range(24):
    estado = "ON " if final_sim['estado_bomba'][t]==True else "OFF"
    print(f"{t+1:4} | {estado:5} | {final_sim['Q_r'][t]:10.6f} | {final_sim['Q_vc_max'][t]:15.6f} | {final_sim['Q_pump'][t]:13.6f} | {final_sim['h'][t]:16.4f}")

print(f"\nAltura final do reservatório: {final_sim['h'][-1]:.4f} m")
print(f"Preço total da energia: {final_sim['fObj']:.2f} €")

# Gráfico de resultados
# Dados do simulador
final_sim = simulador_hidraulico(res_optim.x)
horas = np.arange(1, 25)
nivel_agua = final_sim['h'][:24]  # Nível de água (m)
estado_bomba = final_sim['estado_bomba']  # 1=ON, 0=OFF
energia_kWh = np.array(final_sim['energia_por_hora'])  # Energia (kWh)
custo_acumulado = np.cumsum(final_sim['custo_por_hora'])  # Custo (€)

# Configuração do gráfico
fig, ax1 = plt.subplots(figsize=(14, 7))
plt.title("Monitorização Integrada do Sistema de Bombeamento", fontsize=16, pad=20)

# Eixo principal: Nível de água (azul)
ax1.plot(horas, nivel_agua, 'b-', linewidth=3, label='Nível de água (m)')
ax1.set_xlabel('Hora do Dia', fontsize=12)
ax1.set_ylabel('Nível de Água (m)', color='b', fontsize=12)
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_ylim(0, altura_max + 1)
ax1.set_xticks(horas)
ax1.grid(True, linestyle='--', alpha=0.6)

# Linhas de referência para alturas
ax1.axhline(altura_min, color='r', linestyle=':', linewidth=2, label='Altura Mín (2m)')
ax1.axhline(altura_max, color='g', linestyle=':', linewidth=2, label='Altura Máx (7m)')

# Eixo secundário 1: Estado da bomba (laranja - barras)
ax2 = ax1.twinx()
ax2.bar(horas, estado_bomba, color='orange', alpha=0.3, width=0.8, label='Bomba (ON/OFF)')
ax2.set_ylabel('Estado da Bomba', color='orange', fontsize=12)
ax2.set_ylim(0, 1.5)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['OFF', 'ON'])
ax2.tick_params(axis='y', labelcolor='orange')

# Eixo secundário 2: Energia (kWh - roxo)
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # Eixo externo
ax3.bar(horas, energia_kWh, color='purple', alpha=0.5, width=0.4, label='Energia (kWh)')
ax3.set_ylabel('Energia (kWh)', color='purple', fontsize=12)
ax3.tick_params(axis='y', labelcolor='purple')

# Eixo secundário 3: Custo acumulado (€ - preto)
ax4 = ax1.twinx()
ax4.spines['right'].set_position(('outward', 120))  # Eixo externo
ax4.plot(horas, custo_acumulado, 'k-', marker='s', markersize=8, linewidth=2, label='Custo Acumulado (€)')
ax4.set_ylabel('Custo Acumulado (€)', color='k', fontsize=12)
ax4.tick_params(axis='y', labelcolor='k')

# Legendas organizadas
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax4.get_legend_handles_labels()

# Junta todas as legendas e posiciona à direita
fig.legend(lines1 + lines2 + lines3 + lines4, labels1 + labels2 + labels3 + labels4,
          loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10)

plt.tight_layout(rect=[0, 0, 0.85, 1])  # Ajusta o layout para acomodar a legenda
plt.show()