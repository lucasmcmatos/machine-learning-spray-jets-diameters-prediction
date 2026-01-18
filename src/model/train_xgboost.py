# train_xgboost.py

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit  # split por grupo (sem vazamento)
from sklearn.multioutput import MultiOutputRegressor  # permite múltiplos alvos
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # métricas
from xgboost import XGBRegressor  # modelo XGBoost

# Caminho base do projeto
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_PATH = os.path.join(ROOT_DIR, "data/processed/dataset.csv")
REPORTS_PATH = os.path.join(ROOT_DIR, "reports")
MODELS_PATH = os.path.join(ROOT_DIR, "models")

# Carregar os dados
df = pd.read_csv(DATA_PATH)

# -------------------------------
# Renomeando colunas para evitar erro no XGBoost
# -------------------------------
df = df.rename(columns={
    "Pressure [ atm ]": "pressure_atm",      # renomeia pressão
    "X [ m ]": "x_m",                         # renomeia coordenada X
    "Y [ m ]": "y_m",                         # renomeia coordenada Y
    "Velocity [ m s^-1 ]": "velocity_ms"     # renomeia velocidade (target)
})


# Definir features e targets
features = [
    "pressure_atm",
    "D_in_mm",
    "time_step",
    "x_m",
    "y_m"
]

targets = [
    "current_diameter",
    "velocity_ms"
]


#Criando a simulação por grupo
df["group_id"] = df["simulation_id"]
X = df[features]
y = df[targets]
groups = df["group_id"]


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

# Criar pasta de relatórios
os.makedirs(REPORTS_PATH, exist_ok=True)

# Gráfico de distribuição
plt.figure(figsize=(10, 6))
sns.histplot(df[targets], kde=True)
plt.title("Distribuição do Diâmetro Atual")
plt.xlabel("Diâmetro Atual (mm)")
plt.ylabel("Frequência")
plt.savefig(os.path.join(REPORTS_PATH, "10-xgboost-distribuicao-dataset.png"))
plt.close()
print("Gráfico de distribuição salvo em '10-xgboost-distribuicao-dataset.png'")

# Heatmap de correlação
corrmat = df[features + targets].corr()
k = 4
cols = corrmat.nlargest(k, targets)[targets].index
cm = np.corrcoef(df[cols].T)
plt.figure(figsize=(8, 6))
sns.set(font_scale=1)
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
            annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values)
plt.title("Correlação entre as Top 4 Features e o Diâmetro Atual")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_PATH, "11-xgboost-heatmap-correlacao.png"))
plt.close()
print("Mapa de correlação salvo em '11-xgboost-heatmap-correlacao.png'")

# Treinar modelo
base_xgb = XGBRegressor(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model = MultiOutputRegressor(base_xgb)

print("\nTREINANDO XGBOOST (SEM VAZAMENTO)...")
model.fit(X_train, y_train)

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

print("\nRESULTADOS XGBOOST (GENERALIZAÇÃO REAL):")
print(f"DIÂMETRO  -> RMSE: {rmse_d:.6f} | MAE: {mae_d:.6f} | R²: {r2_d:.6f}")
print(f"VELOCIDADE-> RMSE: {rmse_v:.6f} | MAE: {mae_v:.6f} | R²: {r2_v:.6f}")

import os
MODELS_PATH = os.path.join(ROOT_DIR, "models")
os.makedirs(MODELS_PATH, exist_ok=True)

# Salva o booster do modelo de diâmetro
model.estimators_[0].get_booster().save_model(
    os.path.join(MODELS_PATH, "xgb_diameter_model.json")
)  # salva o modelo XGBoost nativo para diâmetro

# Salva o booster do modelo de velocidade
model.estimators_[1].get_booster().save_model(
    os.path.join(MODELS_PATH, "xgb_velocity_model.json")
)  # salva o modelo XGBoost nativo para velocidade

print("\nMODELOS XGBOOST SALVOS COM SUCESSO (DIÂMETRO E VELOCIDADE)")