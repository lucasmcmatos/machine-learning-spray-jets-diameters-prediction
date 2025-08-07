import argparse
import numpy as np
import joblib
import os
import json
from xgboost import XGBRegressor

# Caminhos absolutos
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_PATH = os.path.join(ROOT_DIR, "models")

# Argumentos da linha de comando
parser = argparse.ArgumentParser(description="Realiza predição manual com o modelo treinado.")
parser.add_argument("--modelo", type=str, required=True,
                    choices=["regressao_linear", "random_forest", "xgboost"],
                    help="Modelo a ser usado para a predição.")
parser.add_argument("--pressao", type=float, required=True, help="Pressão em atm")
parser.add_argument("--velocidade", type=float, required=True, help="Velocidade em m/s")
parser.add_argument("--diametro", type=float, required=True, help="Diâmetro inicial em mm")
parser.add_argument("--time_step", type=int, required=True, help="Time step (passo de tempo)")
args = parser.parse_args()

# Monta o vetor de entrada
entrada = np.array([[args.pressao, args.velocidade, args.diametro, args.time_step]])

# Seleciona e carrega o modelo
if args.modelo == "regressao_linear":
    model_file = os.path.join(MODEL_PATH, "modelo_regressao_linear.npz")
    if not os.path.exists(model_file):
        raise FileNotFoundError("Modelo de Regressão Linear não encontrado.")
    data = np.load(model_file)
    w = data["w"]
    b = data["b"]
    pred = np.dot(entrada, w) + b

elif args.modelo == "random_forest":
    model_file = os.path.join(MODEL_PATH, "modelo_random_forest.pkl")
    if not os.path.exists(model_file):
        raise FileNotFoundError("Modelo Random Forest não encontrado.")
    model = joblib.load(model_file)
    pred = model.predict(entrada)

elif args.modelo == "xgboost":
    model_file = os.path.join(MODEL_PATH, "modelo_xgboost.json")
    if not os.path.exists(model_file):
        raise FileNotFoundError("Modelo XGBoost não encontrado.")
    model = XGBRegressor()
    model.load_model(model_file)
    pred = model.predict(entrada)

else:
    raise ValueError("Modelo inválido selecionado.")

# Resultado
print("\nPARAMETROS DE ENTRADA:")
print(f"Pressão: {args.pressao} atm")
print(f"Velocidade: {args.velocidade} m/s")
print(f"Diâmetro inicial: {args.diametro} mm")
print(f"Time Step: {args.time_step}")
print(f"\nPREDICAO DO DIAMETRO ATUAL: {pred[0]:.4f} mm")
