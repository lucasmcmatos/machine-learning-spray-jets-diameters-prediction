import os
import pandas as pd
import numpy as np
from glob import glob

# Caminhos fixos conforme estrutura do projeto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "dataset.csv")

all_data = []

print("📅 Lendo arquivos da pasta raw...")

for file_path in glob(os.path.join(RAW_DATA_DIR, "*.csv")):
    file_name = os.path.basename(file_path)

    try:
        d_in_mm = float(file_name.replace("d", "").replace("_", ".").replace(".csv", ""))
    except ValueError:
        print(f"⚠️ Nome de arquivo inválido para extração de diâmetro: {file_name}")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Localizar linha de cabeçalho
    header_idx = None
    for idx, line in enumerate(lines):
        if "X" in line and "Y" in line and "Velocity" in line:
            header_idx = idx
            break

    if header_idx is None:
        print(f"❌ Cabeçalho não encontrado em: {file_name}")
        continue

    header = [h.strip() for h in lines[header_idx].strip().split("\t")]
    data_lines = lines[header_idx + 1:]

    cleaned_rows = []
    skip_next = 0
    simulation_id = 1

    for line in data_lines:
        if skip_next > 0:
            skip_next -= 1
            continue

        if line.strip() == "":
            skip_next = 5
            simulation_id += 1
            continue

        values = [v.strip().replace(",", ".") if v.strip() != "null" else None for v in line.strip().split("\t")]
        if len(values) == len(header):
            values.append(simulation_id)
            cleaned_rows.append(values)

    if not cleaned_rows:
        print(f"❌ Nenhum dado válido extraído de {file_name}")
        continue

    try:
        df = pd.DataFrame(cleaned_rows, columns=header + ["simulation"])
        df = df.astype(float)
        df.ffill(inplace=True)
        df["D_in_mm"] = d_in_mm

        # Calcular diâmetro atual para cada linha, baseado na dispersão desde o início da simulação
        current_diameters = []

        for sim_id, group in df.groupby("simulation"):
            y0 = group["Y [ m ]"].iloc[0]
            z0 = group["Z [ m ]"].iloc[0]
            dy = group["Y [ m ]"] - y0
            dz = group["Z [ m ]"] - z0
            d_atual = d_in_mm + 2 * np.sqrt(dy**2 + dz**2)
            current_diameters.extend(d_atual.tolist())

        df["current_diameter"] = current_diameters

        all_data.append(df)
        print(f"✅ Processado: {file_name} (Amostras: {len(df)})")
    except Exception as e:
        print(f"❌ Erro ao processar {file_name}: {e}")

# Salvar dataset final
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    final_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\n✅ Dataset final salvo em: {PROCESSED_DATA_PATH}")
    print(f"📊 Total de amostras: {len(final_df)}")
else:
    print("❌ Nenhum dado válido foi processado. Verifique os arquivos.")
