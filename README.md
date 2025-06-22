# machine-learning-prediction-spray-jets-diameters
# Predição do Diâmetro de Jatos de Combustível com Machine Learning

Este projeto tem como objetivo prever o diâmetro de jatos de combustível em ambientes simulados, utilizando diferentes modelos de aprendizado de máquina. Ele processa dados experimentais provenientes de simulações computacionais para gerar insights sobre o comportamento do jato com base em variáveis como pressão, velocidade e diâmetro inicial.

---

## Estrutura do Projeto

```
├── data/
│   ├── raw/              # Dados brutos (arquivos .csv das simulações)
│   └── processed/        # Dataset tratado e unificado
├── models/               # Modelos treinados salvos
├── reports/              # Relatórios e gráficos gerados
├── src/
│   ├── data/
│   │   └── make_dataset.py           # Geração do dataset final
│   └── model/
│       ├── train_regressao_linear.py
│       ├── train_random_forest.py
│       └── train_xgboost.py
├── Makefile              # Comandos automatizados
├── requirements.txt      # Dependências do projeto
└── README.md             # Documentação principal
```

---

## Como usar

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/seu-repo.git
cd seu-repo
```

### 2. Instale as dependências

Recomenda-se usar um ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Adicione os dados brutos

Coloque todos os arquivos `.csv` das simulações na pasta:

```
data/raw/
```

Cada arquivo deve seguir o formato de nome: `dX_Y.csv`, onde `X_Y` indica o valor do diâmetro (ex: `d2_5.csv` representa 2.5 mm).

---

## Como executar

Utilize o Makefile para executar as etapas do projeto. Os principais comandos são:

### Gerar o dataset processado:

```bash
make criar-dataset
```

### Treinar os modelos:

```bash
make treinar-regressao
make treinar-random-forest
make treinar-xgboost
```

---

## Resultados

- Os gráficos e relatórios gerados durante o treinamento são salvos na pasta `reports/`.
- Os modelos treinados são salvos em `models/` com os nomes apropriados para cada algoritmo.

---

## Modelos Utilizados

- Regressão Linear Regularizada (implementação manual com Gradiente Descendente)
- Random Forest Regressor
- XGBoost Regressor

Cada modelo é avaliado utilizando as seguintes métricas:
- MSE, RMSE, MAE, MAPE, R²

---

## Observações

- O script `make_dataset.py` calcula corretamente o diâmetro atual do jato com base na dispersão em Y e Z, garantindo que as predições sejam fisicamente significativas.
- O pipeline está organizado de forma modular para fácil manutenção, análise e futura extensão do projeto.

---

## Autor

Desenvolvido por Lucas Matos — Projeto acadêmico com aplicação prática em simulações de jatos em dinâmica de fluidos.
