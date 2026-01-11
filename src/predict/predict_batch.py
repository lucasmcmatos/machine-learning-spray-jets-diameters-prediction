# ================================
# PREDIÇÃO EM LOTE - MULTI-OUTPUT
# ================================

import argparse
import pandas as pd
import joblib
import os

# -------------------------------
# Argumentos
# -------------------------------
parser = argparse.ArgumentParser(description="Predição em batch de diâmetro e velocidade.")
parser.add_argument("--modelo", type=str, required=True,
                    choices=["random_forest", "xgboost", "mlp"],
                    help="Modelo a ser usado.")
parser.add_argument("--arquivo", type=str, required=True,
                    help="CSV de entrada em data/prediction/inputs/")
args = parser.parse_args()

# -------------------------------
# Caminhos
# -------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
INPUT_PATH = os.path.join(ROOT_DIR, "data/prediction/inputs", args.arquivo)
OUTPUT_DIR = os.path.join(ROOT_DIR, "data/prediction/outputs")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

MODEL_MAP = {
    "random_forest": "random_forest_multioutput.joblib",
    "xgboost": "xgb_multioutput.joblib",
    "mlp": "mlp_multioutput.joblib"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Verificações
# -------------------------------
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_PATH}")

model_path = os.path.join(MODEL_DIR, MODEL_MAP[args.modelo])
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

# -------------------------------
# Leitura do CSV
# -------------------------------
df = pd.read_csv(INPUT_PATH)

required_cols = ["pressure_atm", "D_in_mm", "time_step", "x_m", "y_m"]

if not all(col in df.columns for col in required_cols):
    raise ValueError(f"O CSV deve conter as colunas: {required_cols}")

X = df[required_cols]

# -------------------------------
# Predição
# -------------------------------
model = joblib.load(model_path)
pred = model.predict(X)

df["predicted_diameter"] = pred[:, 0]
df["predicted_velocity"] = pred[:, 1]

# -------------------------------
# Salvar resultado
# -------------------------------
output_file = os.path.join(
    OUTPUT_DIR,
    f"prediction_{args.modelo}_{args.arquivo}"
)

df.to_csv(output_file, index=False)

print(f"\n✅ Predições salvas com sucesso em:\n{output_file}")
