import os
import argparse
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor

# Argumentos
parser = argparse.ArgumentParser(description="Predição em lote de diâmetro de jatos.")
parser.add_argument("--modelo", type=str, required=True, help="Modelo: 'random_forest', 'xgboost' ou 'regressao_linear'")
parser.add_argument("--arquivo", type=str, required=True, help="Nome do arquivo CSV na pasta data/prediction/inputs/")
args = parser.parse_args()
modelo = args.modelo

# Caminhos
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
INPUT_PATH = os.path.join(ROOT_DIR, "data/prediction/inputs", args.arquivo)

nome_base_saida = os.path.splitext(args.arquivo)[0]
OUTPUT_PATH = os.path.join(ROOT_DIR, "data/prediction/outputs", f"prediction_{nome_base_saida}.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

# Verifica se o arquivo existe
if not os.path.exists(INPUT_PATH):
    print(f"❌ Arquivo de entrada não encontrado: {INPUT_PATH}")
    exit(1)

# Carrega os dados
df = pd.read_csv(INPUT_PATH)

# Features esperadas
features = ["Pressure [ atm ]", "Velocity [ m s^-1 ]", "D_in_mm", "time_step"]

# Verificação de colunas
if not all(col in df.columns for col in features):
    print("❌ O arquivo CSV deve conter as seguintes colunas:")
    print(features)
    exit(1)

if modelo == "xgboost":
    df = df.rename(columns={
        "Pressure [ atm ]": "Pressure",
        "Velocity [ m s^-1 ]": "Velocity",
        "D_in_mm": "D_in"
    })

    features = ["Pressure","Velocity","D_in","time_step"]

X = df[features]

# Carregamento do modelo
modelo = args.modelo.lower()

if modelo == "random_forest":
    model_path = os.path.join(MODEL_DIR, "modelo_random_forest.pkl")
    model = joblib.load(model_path)

elif modelo == "xgboost":
    model_path = os.path.join(MODEL_DIR, "modelo_xgboost.json")
    model = XGBRegressor()
    model.load_model(model_path)

elif modelo == "regressao_linear":
    model_path = os.path.join(MODEL_DIR, "modelo_regressao_linear.npz")
    data = np.load(model_path)
    w = data["w"]
    b = data["b"]
else:
    print("❌ Modelo inválido. Use: random_forest, xgboost ou regressao_linear")
    exit(1)

# Realizando a predição
if modelo == "regressao_linear":
    predictions = np.dot(X, w) + b
else:
    predictions = model.predict(X)

# Salvando resultado
df["predicted_diameter"] = predictions
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Predições salvas com sucesso em: {OUTPUT_PATH}")
