# train_random_forest.py

# Importando bibliotecas necessárias
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

# Separação das features e do alvo
features = ["Pressure [ atm ]", "Velocity [ m s^-1 ]", "D_in_mm", "time_step"]
target = "current_diameter"
X = df[features]
y = df[target]

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nINFORMAÇÕES DOS DADOS:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# ---------- Gráfico 06: Distribuição do target ----------
plt.figure(figsize=(10, 6))
sns.histplot(y, kde=True)
plt.title("Distribuição do Diâmetro Atual (Random Forest)")
plt.xlabel("Diâmetro Atual (mm)")
plt.ylabel("Frequência")
plt.savefig(os.path.join(REPORT_PATH, "06-rf-distribuicao-dataset.png"))
plt.close()
print("Gráfico de distribuição salvo em '06-rf-distribuicao-dataset.png'")

# ---------- Gráfico 07: Correlação ----------
corrmat = df[features + [target]].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corrmat, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mapa de Correlação - Random Forest")
plt.savefig(os.path.join(REPORT_PATH, "07-rf-heatmap-correlacao.png"))
plt.close()
print("Mapa de correlação salvo em '07-rf-heatmap-correlacao.png'")

# ---------- Treinamento do modelo Random Forest ----------
print("\nTREINAMENTO DO MODELO RANDOM FOREST...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Treinamento concluído com sucesso!")

# ---------- Avaliação ----------
print("\nREALIZANDO AVALIAÇÃO DO MODELO...")

print("\nAMOSTRAS DE PREDIÇÕES VS VALORES REAIS:")
# Predições
predictions = rf.predict(X_test)

for i in range(10):  # mostra só as 10 primeiras por simplicidade
    print(f"Amostra {i+1}: Predição = {predictions[i]:.4f}, Valor real = {y_test.iloc[i]:.4f}")

# Métricas
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

print("\nANALISE DA CORRELACAO ENTRE TIME_STEP E O TARGET:")
print(df[["time_step", "current_diameter"]].corr())

print("\nMÉTRICAS DE AVALIAÇÃO:")
print(f"MSE : {mse:.8f}")
print(f"RMSE: {rmse:.8f}")
print(f"R²  : {r2:.8f}")
print(f"MAE : {mae:.8f}")
print(f"MAPE: {mape:.8f}%")

# ---------- Gráfico 08: Comparação predição vs valor real ----------
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Valores Reais", marker='o')
plt.plot(predictions, label="Previsões RF", marker='x')
plt.legend()
plt.title("Comparação entre Previsões e Valores Reais - RF")
plt.xlabel("Amostras de Teste")
plt.ylabel("Diâmetro Atual (mm)")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_PATH, "08-rf-comparacao-predicao-real.png"))
plt.close()
print("Gráfico comparativo salvo em '08-rf-comparacao-predicao-real.png'")

# ---------- Importância das Features (Gráfico 09) ----------
importances = rf.feature_importances_
plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=features)
plt.title("Importância das Features - Random Forest")
plt.xlabel("Importância")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_PATH, "09-rf-importancia-features.png"))
plt.close()
print("Gráfico de importância salvo em '09-rf-importancia-features.png'")

# ---------- Salvando o modelo ----------
joblib.dump(rf, os.path.join(MODEL_PATH, "modelo_random_forest.pkl"))
print("Modelo Random Forest salvo em '../../models/modelo_random_forest.pkl'")
