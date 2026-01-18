import os                          # Manipulação de caminhos e diretórios
import pandas as pd               # Estrutura de dados tabular
import numpy as np                # Operações numéricas
from glob import glob              # Listagem de arquivos por padrão

# ===============================
# Definição de caminhos do projeto
# ===============================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # Raiz do projeto
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")                             # Pasta com CSVs brutos
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "dataset.csv") # Dataset final

all_data = []                 # Lista para armazenar DataFrames de cada arquivo
global_simulation_id = 0      # Contador global de simulações (não reinicia por arquivo)

print("📅 Lendo arquivos da pasta raw...")

# ===============================
# Leitura de cada CSV bruto
# ===============================
for file_path in glob(os.path.join(RAW_DATA_DIR, "*.csv")):
    file_name = os.path.basename(file_path)

    # -------------------------------
    # Extração do diâmetro inicial a partir do nome do arquivo
    # Exemplo: d2_8.csv -> 2.8 mm
    # -------------------------------
    try:
        d_in_mm = float(file_name.replace("d", "").replace("_", ".").replace(".csv", ""))
    except ValueError:
        print(f"⚠️ Nome de arquivo inválido para extração de diâmetro: {file_name}")
        continue

    # -------------------------------
    # Leitura bruta do arquivo (linha a linha)
    # -------------------------------
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # -------------------------------
    # Localizar linha de cabeçalho
    # -------------------------------
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
    local_simulation_id = 0   # ID local apenas para detectar quebras de simulação

    # -------------------------------
    # Limpeza e identificação das simulações
    # -------------------------------
    for line in data_lines:
        if skip_next > 0:
            skip_next -= 1
            continue

        # Linha vazia indica nova simulação
        if line.strip() == "":
            skip_next = 5
            local_simulation_id += 1
            global_simulation_id += 1
            continue

        # Ignora linhas que claramente usam ';' como separador (blocos inválidos do solver)
        if ";" in line:
            continue  # pula linha inválida

        raw_values = line.strip().split("\t")  # separa por TAB (formato válido esperado)

        # Se o número de colunas não bate com o cabeçalho, a linha é descartada
        if len(raw_values) != len(header):
            continue  # evita linhas corrompidas ou metadados

        # Limpeza e conversão preliminar dos valores
        values = []
        linha_valida = True

        for v in raw_values:
            v = v.strip().replace(",", ".")
            if v.lower() == "null" or v == "":
                values.append(np.nan)
            else:
                try:
                    values.append(float(v))
                except ValueError:
                    linha_valida = False
                    break  # se não for número, descarta a linha inteira

        # Apenas adiciona se todos os valores forem numéricos
        if linha_valida:
            values.append(global_simulation_id)
            cleaned_rows.append(values)

    if not cleaned_rows:
        print(f"❌ Nenhum dado válido extraído de {file_name}")
        continue

    # ===============================
    # Construção do DataFrame
    # ===============================
    df = pd.DataFrame(cleaned_rows, columns=header + ["simulation_id"])
    # Converte apenas colunas numéricas, ignora problemas silenciosamente
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove linhas que ainda ficaram inválidas
    df.dropna(inplace=True)          # Converte tudo para float
    df.ffill(inplace=True)          # Preenchimento forward-fill

    # -------------------------------
    # Atribuição explícita do diâmetro inicial
    # (constante por simulação)
    # -------------------------------
    df["D_in_mm"] = d_in_mm

    # -------------------------------
    # Renomeio semântico da pressão
    # Mantemos pressure_atm por compatibilidade
    # -------------------------------
    if "Pressure" in df.columns:
        df["pressure_norm"] = df["Pressure"]      # Cópia explícita
        df.rename(columns={"Pressure": "pressure_atm"}, inplace=True)

    # -------------------------------
    # Cálculo do diâmetro atual
    # Base geométrica: expansão radial no plano Y-Z
    # -------------------------------
    current_diameters = []
    time_steps_all = []

    for sim_id, group in df.groupby("simulation_id"):
        y0 = group["Y [ m ]"].iloc[0]
        z0 = group["Z [ m ]"].iloc[0]

        dy = group["Y [ m ]"] - y0
        dz = group["Z [ m ]"] - z0

        # Diâmetro atual: diâmetro inicial + expansão radial
        d_atual = d_in_mm + 2 * np.sqrt(dy**2 + dz**2)

        time_steps = np.arange(len(group))

        current_diameters.extend(d_atual.tolist())
        time_steps_all.extend(time_steps.tolist())

    df["current_diameter"] = current_diameters
    df["time_step"] = time_steps_all

    all_data.append(df)
    print(f"✅ Processado: {file_name} (Amostras: {len(df)})")

# ===============================
# Salvamento do dataset final
# ===============================
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    final_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\n✅ Dataset final salvo em: {PROCESSED_DATA_PATH}")
    print(f"📊 Total de amostras: {len(final_df)}")
else:
    print("❌ Nenhum dado válido foi processado. Verifique os arquivos.")
