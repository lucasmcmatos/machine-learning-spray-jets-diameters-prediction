import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_PATH = os.path.join(ROOT_DIR, "data/processed/dataset.csv")
REPORT_PATH = os.path.join(ROOT_DIR, "reports")
MODEL_PATH = os.path.join(ROOT_DIR, "models")

os.makedirs(REPORT_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

df = pd.read_csv(DATA_PATH)


df = df.rename(columns={
    "Pressure [ atm ]": "pressure_atm",
    "X [ m ]": "x_m",
    "Y [ m ]": "y_m",
    "Velocity [ m s^-1 ]": "velocity_ms"
})

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

df["group_id"] = (
    df["simulation"].astype(str) + "_" +
    df["D_in_mm"].astype(str)
)

X = df[features]
y = df[targets]
groups = df["group_id"]

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train = X.iloc[train_idx]
X_test  = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test  = y.iloc[test_idx]

print("\nINFORMAÇÕES DOS DADOS:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test : {X_test.shape}, y_test : {y_test.shape}")

mlp = MLPRegressor(
    hidden_layer_sizes=(64, 64, 32),   # arquitetura
    activation="relu",
    solver="adam",
    alpha=1e-4,                        # regularização L2
    batch_size=256,
    learning_rate_init=1e-3,
    max_iter=500,
    early_stopping=True,
    random_state=42
)

pipeline = Pipeline([
    ("scaler_X", StandardScaler()),    # normaliza entradas
    ("mlp", mlp)
])

model = MultiOutputRegressor(pipeline)

print("\nTREINANDO MLP (SEM VAZAMENTO)...")
model.fit(X_train, y_train)

pred = model.predict(X_test)

pred_diam = pred[:, 0]
pred_vel  = pred[:, 1]

y_test_diam = y_test["current_diameter"].values
y_test_vel  = y_test["velocity_ms"].values

rmse_d = np.sqrt(mean_squared_error(y_test_diam, pred_diam))
mae_d  = mean_absolute_error(y_test_diam, pred_diam)
r2_d   = r2_score(y_test_diam, pred_diam)

rmse_v = np.sqrt(mean_squared_error(y_test_vel, pred_vel))
mae_v  = mean_absolute_error(y_test_vel, pred_vel)
r2_v   = r2_score(y_test_vel, pred_vel)

print("\nRESULTADOS MLP (GENERALIZAÇÃO REAL):")
print(f"DIÂMETRO  -> RMSE: {rmse_d:.6f} | MAE: {mae_d:.6f} | R²: {r2_d:.6f}")
print(f"VELOCIDADE-> RMSE: {rmse_v:.6f} | MAE: {mae_v:.6f} | R²: {r2_v:.6f}")

joblib.dump(
    model,
    os.path.join(MODEL_PATH, "mlp_multioutput.joblib")
)

print("\nMODELO MLP SALVO COM SUCESSO!")