# Makefile para treinar modelos de predição de diâmetro de jatos

.PHONY: todos criar-dataset treinar-regressao-linear treinar-randomforest treinar-xgboost limpar

# Caminho base para os scripts
SCRIPTS=src/model

# Comando para rodar tudo (gerar base + treinar modelos)
todos: criar-dataset criar-dataset treinar-regressao-linear treinar-randomforest treinar-xgboost

# Geração da base de dados
criar-dataset:
	@echo "Gerando base de dados..."
	python3 src/data/make_dataset.py

# Regressão Linear
treinar-regressao-linear:
	@echo "Executando treinamento: Regressão Linear..."
	python3 $(SCRIPTS)/train_mult_linear.py

# Random Forest
treinar-randomforest:
	@echo "Executando treinamento: Random Forest..."
	python3 $(SCRIPTS)/train_random_forest.py

# XGBoost
treinar-xgboost:
	@echo "Executando treinamento: XGBoost..."
	python3 $(SCRIPTS)/train_xgboost.py

# Limpar arquivos de relatório e modelos
limpar:
	@echo "Limpando relatórios e modelos salvos..."
	rm -rf reports/*.png
	rm -rf models/*

# Predição manual com modelo treinado
prever-manual:
	@echo "Executando predição manual..."
	python3 src/predict/predict_manual.py \
		--modelo $(modelo) \
		--pressao $(pressao) \
		--velocidade $(velocidade) \
		--diametro $(diametro) \
		--time_step $(time_step)

# Executa predição em lote a partir de um arquivo CSV
# Exemplo: make prever-batch MODELO=xgboost ARQUIVO=exemplo.csv
prever-batch:
	@echo "Executando predição em lote com arquivo: $(ARQUIVO) usando modelo: $(MODELO)..."
	python3 src/predict/predict_batch.py --modelo $(MODELO) --arquivo $(ARQUIVO)