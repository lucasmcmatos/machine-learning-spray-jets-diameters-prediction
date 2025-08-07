# train_xgboost.py

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Caminho base do projeto
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_PATH = os.path.join(ROOT_DIR, "data/processed/dataset.csv")
REPORTS_PATH = os.path.join(ROOT_DIR, "reports")
MODELS_PATH = os.path.join(ROOT_DIR, "models")

# Carregar os dados
df = pd.read_csv(DATA_PATH)

# Definir features e target
features = ["Pressure [ atm ]", "Velocity [ m s^-1 ]", "D_in_mm","time_step"]
target = "current_diameter"
X = np.array(df[features])
y = np.array(df[target])

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nINFORMAÇÕES DOS DADOS:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# Criar pasta de relatórios
os.makedirs(REPORTS_PATH, exist_ok=True)

# Gráfico de distribuição
plt.figure(figsize=(10, 6))
sns.histplot(df[target], kde=True)
plt.title("Distribuição do Diâmetro Atual")
plt.xlabel("Diâmetro Atual (mm)")
plt.ylabel("Frequência")
plt.savefig(os.path.join(REPORTS_PATH, "10-xgboost-distribuicao-dataset.png"))
plt.close()
print("Gráfico de distribuição salvo em '10-xgboost-distribuicao-dataset.png'")

# Heatmap de correlação
corrmat = df[features + [target]].corr()
k = 4
cols = corrmat.nlargest(k, target)[target].index
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

# Renomear colunas para o XGBoost
X_train_df = pd.DataFrame(X_train, columns=["Pressure", "Velocity", "D_in", "time_step"])
X_test_df = pd.DataFrame(X_test, columns=["Pressure", "Velocity", "D_in", "time_step"])

# Treinar modelo
print("\nTREINAMENTO DO MODELO XGBOOST...")
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb.fit(X_train_df, y_train)

# Predições
predictions = xgb.predict(X_test_df)

# Métricas
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

print("\nAMOSTRAS DE PREDIÇÕES VS VALORES REAIS:")
for i in range(10):
    print(f"Amostra {i+1}: Predição = {predictions[i]:.4f}, Valor real = {y_test[i]:.4f}")


# Resultados
print("\nMÉTRICAS DE AVALIAÇÃO DO MODELO XGBOOST:")
print(f"MSE : {mse:.8f}")
print(f"RMSE: {rmse:.8f}")
print(f"R²  : {r2:.8f}")
print(f"MAE : {mae:.8f}")
print(f"MAPE: {mape:.8f}%")

# Salvar modelo
print("\nSALVANDO O MODELO XGBOOST...")
os.makedirs(MODELS_PATH, exist_ok=True)
xgb.save_model(os.path.join(MODELS_PATH, "modelo_xgboost.json"))
print("Modelo salvo com sucesso em '../../models/modelo_xgboost.json'")
