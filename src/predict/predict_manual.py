# ================================
# PREDIÇÃO MANUAL - MULTI-OUTPUT
# ================================

import argparse
import numpy as np
import pandas as pd
import joblib
import os
import xgboost as xgb

# -------------------------------
# Argumentos
# -------------------------------
parser = argparse.ArgumentParser(description="Predição manual de diâmetro e velocidade.")
parser.add_argument("--modelo", type=str, required=True,
                    choices=["random_forest", "xgboost", "mlp"],
                    help="Modelo a ser usado.")
parser.add_argument("--pressao", type=float, required=True, help="Pressão (atm)")
parser.add_argument("--diametro_inicial", type=float, required=True, help="Diâmetro inicial (mm)")
parser.add_argument("--time_step", type=int, required=True, help="Time step")
parser.add_argument("--x", type=float, required=True, help="Coordenada X (m)")
parser.add_argument("--y", type=float, required=True, help="Coordenada Y (m)")
args = parser.parse_args()

# -------------------------------
# Caminhos
# -------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_DIR = os.path.join(ROOT_DIR, "models")

MODEL_MAP = {
    "random_forest": "random_forest_multioutput.joblib",
    "xgboost": "xgboost",
    "mlp": "mlp_multioutput.joblib"
}


if MODEL_MAP[args.modelo] == "xgboost":
    # Caminhos dos modelos XGBoost salvos
    diameter_model_path = os.path.join(MODEL_DIR, "xgb_diameter_model.json")  # modelo do diâmetro
    velocity_model_path = os.path.join(MODEL_DIR, "xgb_velocity_model.json")  # modelo da velocidade

    # Verificação de existência
    if not os.path.exists(diameter_model_path):
        raise FileNotFoundError(f"Modelo de diâmetro não encontrado: {diameter_model_path}")

    if not os.path.exists(velocity_model_path):
        raise FileNotFoundError(f"Modelo de velocidade não encontrado: {velocity_model_path}")

    # -------------------------------
    # Entrada de dados (DataFrame)
    # -------------------------------
    X = pd.DataFrame([{
        "pressure_atm": args.pressao,         # pressão (atm)
        "D_in_mm": args.diametro_inicial,      # diâmetro inicial (mm)
        "time_step": args.time_step,           # passo de tempo
        "x_m": args.x,                         # coordenada X (m)
        "y_m": args.y                          # coordenada Y (m)
    }])

    # Converter para DMatrix (formato nativo do XGBoost)
    dmatrix = xgb.DMatrix(X)

    # -------------------------------
    # Carregar modelos
    # -------------------------------
    model_diameter = xgb.Booster()  # booster do diâmetro
    model_diameter.load_model(diameter_model_path)  # carrega modelo salvo

    model_velocity = xgb.Booster()  # booster da velocidade
    model_velocity.load_model(velocity_model_path)  # carrega modelo salvo

    # -------------------------------
    # Predições
    # -------------------------------
    pred_diameter = model_diameter.predict(dmatrix)[0]  # predição do diâmetro
    pred_velocity = model_velocity.predict(dmatrix)[0]  # predição da velocidade

    # -------------------------------
    # Saída
    # -------------------------------
    print("\nPARÂMETROS DE ENTRADA:")
    print(X.to_string(index=False))

    print("\nRESULTADO DA PREDIÇÃO (XGBOOST):")
    print(f"Diâmetro previsto  : {pred_diameter:.6f}")
    print(f"Velocidade prevista: {pred_velocity:.6f}")
    
else:
    model_path = os.path.join(MODEL_DIR, MODEL_MAP[args.modelo])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    # -------------------------------
    # Entrada (DataFrame)
    # -------------------------------
    X = pd.DataFrame([{
        "pressure_atm": args.pressao,
        "D_in_mm": args.diametro_inicial,
        "time_step": args.time_step,
        "x_m": args.x,
        "y_m": args.y
    }])

    # -------------------------------
    # Predição
    # -------------------------------
    model = joblib.load(model_path)
    pred = model.predict(X)

    diametro_pred = pred[0, 0]
    velocidade_pred = pred[0, 1]

    # -------------------------------
    # Saída
    # -------------------------------
    print("\nPARÂMETROS DE ENTRADA:")
    print(X.to_string(index=False))

    print("\nRESULTADO DA PREDIÇÃO:")
    print(f"Diâmetro previsto  : {diametro_pred:.6f}")
    print(f"Velocidade prevista: {velocidade_pred:.6f}")
