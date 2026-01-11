# train_random_forest.py

# Importando bibliotecas necessárias
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit  # split por grupos (sem vazamento)
from sklearn.multioutput import MultiOutputRegressor   # suporte a múltiplos alvos
from sklearn.ensemble import RandomForestRegressor     # Random Forest
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Definindo caminhos absolutos dinâmicos
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_PATH = os.path.join(ROOT_DIR, "data/processed/dataset.csv")
REPORT_PATH = os.path.join(ROOT_DIR, "reports")
MODEL_PATH = os.path.join(ROOT_DIR, "models")
os.makedirs(REPORT_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# Carregando os dados
df = pd.read_csv(DATA_PATH)

df = df.rename(columns={
    "Pressure [ atm ]": "pressure_atm",      # pressão
    "X [ m ]": "x_m",                         # coordenada X
    "Y [ m ]": "y_m",                         # coordenada Y
    "Velocity [ m s^-1 ]": "velocity_ms"     # velocidade (targets)
})

# Separação das features e do alvo
features = [
    "pressure_atm",  # pressão do sistema
    "D_in_mm",       # diâmetro inicial
    "time_step",     # instante de tempo
    "x_m",           # coordenada espacial X
    "y_m"            # coordenada espacial Y
]

# -------------------------------
# Definição dos TARGETsS (y)
# -------------------------------
targets = [
    "current_diameter",  # diâmetro no timestep t
    "velocity_ms"        # velocidade no timestep t
]

df["group_id"] = (
    df["simulation"].astype(str) + "_" +
    df["D_in_mm"].astype(str)
)

X = df[features]
y = df[targets]
groups = df["group_id"]   # grupos para split correto


# Divisão treino/teste
gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test  = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test  = y.iloc[test_idx]

print("\nINFORMAÇÕES DOS DADOS:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# ---------- Gráfico 06: Distribuição do targets ----------
plt.figure(figsize=(10, 6))
sns.histplot(y, kde=True)
plt.title("Distribuição do Diâmetro Atual (Random Forest)")
plt.xlabel("Diâmetro Atual (mm)")
plt.ylabel("Frequência")
plt.savefig(os.path.join(REPORT_PATH, "06-rf-distribuicao-dataset.png"))
plt.close()
print("Gráfico de distribuição salvo em '06-rf-distribuicao-dataset.png'")

# ---------- Gráfico 07: Correlação ----------
corrmat = df[features + targets].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corrmat, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mapa de Correlação - Random Forest")
plt.savefig(os.path.join(REPORT_PATH, "07-rf-heatmap-correlacao.png"))
plt.close()
print("Mapa de correlação salvo em '07-rf-heatmap-correlacao.png'")

# ---------- Treinamento do modelo Random Forest ----------
print("\nTREINAMENTO DO MODELO RANDOM FOREST...")
base_rf = RandomForestRegressor(
    n_estimators=300,      # número de árvores
    max_depth=None,        # árvores crescem até parar
    random_state=42,
    n_jobs=-1
)

model = MultiOutputRegressor(base_rf)

model.fit(X_train, y_train)
print("Treinamento concluído com sucesso!")

# ---------- Avaliação ----------
print("\nREALIZANDO AVALIAÇÃO DO MODELO...")

print("\nAMOSTRAS DE PREDIÇÕES VS VALORES REAIS:")
# Predições
pred = model.predict(X_test)

pred_diam = pred[:, 0]  # predição do diâmetro
pred_vel  = pred[:, 1]  # predição da velocidade

y_test_diam = y_test["current_diameter"].values
y_test_vel  = y_test["velocity_ms"].values

# Métricas - Diâmetro
rmse_d = np.sqrt(mean_squared_error(y_test_diam, pred_diam))
mae_d  = mean_absolute_error(y_test_diam, pred_diam)
r2_d   = r2_score(y_test_diam, pred_diam)

# Métricas - Velocidade
rmse_v = np.sqrt(mean_squared_error(y_test_vel, pred_vel))
mae_v  = mean_absolute_error(y_test_vel, pred_vel)
r2_v   = r2_score(y_test_vel, pred_vel)

print("\nRESULTADOS RANDOM FOREST (GENERALIZAÇÃO REAL):")
print(f"DIÂMETRO  -> RMSE: {rmse_d:.6f} | MAE: {mae_d:.6f} | R²: {r2_d:.6f}")
print(f"VELOCIDADE-> RMSE: {rmse_v:.6f} | MAE: {mae_v:.6f} | R²: {r2_v:.6f}")


# ---------- Gráfico 08: Comparação predição vs valor real ----------
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Valores Reais", marker='o')
plt.plot(pred, label="Previsões RF", marker='x')
plt.legend()
plt.title("Comparação entre Previsões e Valores Reais - RF")
plt.xlabel("Amostras de Teste")
plt.ylabel("Diâmetro Atual (mm)")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_PATH, "08-rf-comparacao-predicao-real.png"))
plt.close()
print("Gráfico comparativo salvo em '08-rf-comparacao-predicao-real.png'")

# ---------- Importância das Features (Gráfico 09) ----------
importances = model.estimators_[0].feature_importances_

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=features)
plt.title("Importância das Features - Random Forest (Diâmetro)")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_PATH, "rf-importancia-features-diametro.png"))
plt.close()
# ---------- Salvando o modelo ----------
joblib.dump(
    model,
    os.path.join(MODEL_PATH, "random_forest_multioutput.joblib")
)

print("\nMODELO RANDOM FOREST SALVO COM SUCESSO!")