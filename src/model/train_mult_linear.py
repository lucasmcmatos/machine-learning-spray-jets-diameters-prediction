# Importando todas as bibliotecas necessárias
import pandas as pd
import numpy as np
import copy, math
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm
from scipy import stats
from scipy.stats import skew
import os

# Caminhos absolutos dinâmicos
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_PATH = os.path.join(ROOT_DIR, "data/processed/dataset.csv")
REPORT_PATH = os.path.join(ROOT_DIR, "reports")
MODEL_PATH = os.path.join(ROOT_DIR, "models")
os.makedirs(REPORT_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# Instanciando o dataset
df = pd.read_csv(DATA_PATH)

# Separação das features e do target para o projeto.
features = ["Pressure [ atm ]", "Velocity [ m s^-1 ]", "D_in_mm", "time_step"]
target = "current_diameter"
all_data = df[features + ["current_diameter"]]
X = np.array(df[features])
y = np.array(df[target])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nINFORMAÇÕES DOS DADOS:")
print(f"Formato de 'X_train': {X_train.shape}, Tipo de 'X_train':{type(X_train)})")
print(f"Formato de 'y_train': {y_train.shape}, Tipo de 'y_train':{type(y_train)})")
print(f"Formato de 'X_test': {X_test.shape}, Tipo de 'X_test':{type(X_test)})")
print(f"Formato de 'y_test': {y_test.shape}, Tipo de 'y_test':{type(y_test)})")

# Verificando a distribuição dos dados
plt.figure(figsize=(10, 6))
sns.histplot(all_data["current_diameter"], kde=True)
plt.title("Distribuição do Diâmetro Atual")
plt.xlabel("Diâmetro Atual (mm)")
plt.ylabel("Frequência")
plt.savefig(os.path.join(REPORT_PATH, "01-regressao-linear-distribuicao-dataset.png"))
plt.close()
print("\nVERIFICANDO A DISTRIBUIÇÃO DOS DADOS:")
print("Gráfico de distribuição salvo em '01-regressao-linear-distribuicao-dataset.png'")

# Verificando a correlação entre os dados
corrmat = all_data.corr()
k = 4
cols = corrmat.nlargest(k, "current_diameter")["current_diameter"].index
cm = np.corrcoef(all_data[cols].T)
plt.figure(figsize=(8, 6))
sns.set(font_scale=1)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                 annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values)
plt.title("Correlação entre as Top 4 Features e o Diâmetro Atual")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_PATH, "02-regressao-linear-heatmap-correlacao.png"))
plt.close()
print("\nVERIFICANDO A CORRELAÇÃO ENTRE OS DADOS:")
print("Heatmap salvo em '02-regressao-linear-heatmap-correlacao.png'")

# Verificando a dispersão entre as top features e o diâmetro atual
print("\nVERIFICANDO A DISPERSÃO DAS FEATURES EM RELAÇÃO AO DIÂMETRO:")
X_disp = all_data[cols].drop(columns=['current_diameter'])
y_disp = all_data[cols]['current_diameter']
X_train_disp, X_test_disp, y_train_disp, y_test_disp = train_test_split(X_disp, y_disp, test_size=0.2, random_state=42)
X_train_disp = np.array(X_train_disp)
X_test_disp = np.array(X_test_disp)
y_train_disp = np.array(y_train_disp)
y_test_disp = np.array(y_test_disp)
n_features = len(cols) - 1
n_cols = 3
n_rows = int(np.ceil(n_features / n_cols))
fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharey=True)
ax = ax.flatten()
for i in range(n_features):
    ax[i].scatter(X_train_disp[:, i], y_train_disp, alpha=0.6)
    ax[i].set_xlabel(cols[i])
ax[0].set_ylabel("Diâmetro Atual")
for i in range(n_features, len(ax)):
    fig.delaxes(ax[i])
plt.tight_layout()
plt.savefig(os.path.join(REPORT_PATH, "03-regressao-linear-dispersao-features.png"))
plt.close()
print("Gráfico de dispersão salvo em '03-regressao-linear-dispersao-features.png'")

# Inicializando os parâmetros da função de custo
print("\nINICIALIZANDO OS PARAMETROS DA FUNÇÃO DE CUSTO:")
b_init = 0.0
w_init = np.zeros(len(features))
print(f"w_init shape: {w_init.shape}, tipo: {type(w_init)}")
print(f"b_init valor: {b_init}, tipo: {type(b_init)}")

def compute_cost_regularized(X, y, w, b, lambda_=0.7):
    m = X.shape[0]
    n = len(w)
    cost = 0.0
    for i in range(m):
      f_wb_i = np.dot(X[i], w) + b
      cost = cost + (f_wb_i - y[i])**2
    cost  = cost / (2*m)
    reg_cost = 0
    for j in range(n):
      reg_cost += (w[j]**2)
    reg_cost = (lambda_ / (2*m)) * reg_cost
    total_cost = cost + reg_cost
    return total_cost

cost = compute_cost_regularized(X_train, y_train, w_init, b_init)
print("\nCALCULANDO O CUSTO INICIAL:")
print(f'Cost at optimal w : {cost:.4f}')

def compute_gradient_regularized(X, y, w, b, lambda_=0.7):
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):
      err = (np.dot(X[i], w) + b) - y[i]
      for j in range(n):
        dj_dw[j] = dj_dw[j] + err * X[i,j]
      dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    for j in range(n):
      dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]
    return dj_db, dj_dw

def gradient_descent_regularized(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_=0.7):
  J_history = []
  w = copy.deepcopy(w_in)
  b = b_in
  for i in range(num_iters):
        dj_db,dj_dw = gradient_function(X, y, w, b, lambda_)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i<100000:
            J_history.append(cost_function(X, y, w, b, lambda_))
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteração {i:4d}: Custo atual {J_history[-1]:8.4f}")
  return w, b, J_history

tmp_dj_db, tmp_dj_dw = compute_gradient_regularized(X_train, y_train, w_init, b_init)
print(f'dj_db para w,b inicial: {tmp_dj_db}')
print(f'dj_dw para w,b inicial: {tmp_dj_dw}')

# Treinando o modelo
print("\nTREINAMENTO DO MODELO DE PREDIÇÃO")
initial_w = np.zeros_like(w_init)
initial_b = 0.
iterations = 10000
alpha = 5.0e-4
lambda_ = 0.6
w_final, b_final, J_hist = gradient_descent_regularized(X_train, y_train, initial_w, initial_b, compute_cost_regularized, compute_gradient_regularized, alpha, iterations, lambda_)
print(f"\nParâmetros encontrados para o modelo:\n b: {b_final:0.4f}\n w: {w_final} ")

# Gráfico da taxa de aprendizado
print("\nGERANDO GRÁFICO DE AVALIAÇÃO DA TAXA DE APRENDIZADO...")
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax1.set_title("Custo vs Iterações (completo)")
ax1.set_ylabel("Custo")
ax1.set_xlabel("Iterações")
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax2.set_title("Custo vs Iterações (foco após 100)")
ax2.set_ylabel("Custo")
ax2.set_xlabel("Iterações")
plt.savefig(os.path.join(REPORT_PATH, "05-regressao-linear-avaliacao-taxa-aprendizado.png"))
plt.close()
print("Gráfico da taxa de aprendizado salvo em '05-regressao-linear-avaliacao-taxa-aprendizado.png'")

# Predição com o modelo
print("\nREALIZANDO A PREDIÇÃO COM O MODELO FINAL...")
m, _ = X_test.shape
predictions = np.zeros(m)
for i in range(m):
    predictions[i] = np.dot(X_test[i], w_final) + b_final
    print(f"Predição: {predictions[i]:.6f}, Valor real: {y_test[i]:.6f}")

# Avaliação
print("\nMÉTRICAS DE AVALIAÇÃO DO MODELO:")
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")
print(f"MAE : {mae:.4f}")
print(f"MAPE: {mape:.4f}%")

# Salvando os parâmetros do modelo
print("\nSALVANDO O MODELO TREINADO...")
np.savez(os.path.join(MODEL_PATH, "modelo_regressao_linear.npz"), w=w_final, b=b_final)
print("Modelo salvo com sucesso em '../../models/modelo_regressao_linear.npz'")
