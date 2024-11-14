# 變數定義
PYTHON := python
PIP := pip
SRC_DIR := src
DATA_DIR := data
MODELS_DIR := models
RESULTS_DIR := results

# 創建必要的目錄
.PHONY: init
init:
	mkdir -p $(DATA_DIR)/raw $(DATA_DIR)/processed $(MODELS_DIR) $(RESULTS_DIR)

# 安裝依賴
.PHONY: install
install:
	$(PIP) install -r requirements.txt

# 訓練模型
.PHONY: train
train: 
	$(PYTHON) $(SRC_DIR)/train_model.py

# 執行預測
.PHONY: predict
predict:
	$(PYTHON) $(SRC_DIR)/predict.py

# 可視化結果
.PHONY: visualize
visualize:
	$(PYTHON) $(SRC_DIR)/visualize_timeseries.py

# 清理生成的文件
.PHONY: clean
clean:
	rm -rf $(MODELS_DIR)/*
	rm -rf $(RESULTS_DIR)/*
	rm -rf $(DATA_DIR)/processed/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# 執行完整流程
.PHONY: all
all: install preprocess train

# 顯示幫助信息
.PHONY: help
help:
	@echo "可用的命令："
	@echo "  make init       - 創建必要的目錄"
	@echo "  make install    - 安裝所需的 Python 套件"
	@echo "  make preprocess - 執行數據預處理"
	@echo "  make train      - 訓練模型"
	@echo "  make predict    - 執行預測"
	@echo "  make visualize  - 可視化結果"
	@echo "  make clean      - 清理生成的文件"
	@echo "  make all        - 執行完整流程（安裝、預處理、訓練）" 