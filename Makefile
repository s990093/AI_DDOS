# 基本變數設定
PYTHON = python
SRC_DIR = src
SCRIPT = $(SRC_DIR)/main.py

# 默認參數
CORES = -1
ITERATIONS = 300
POPULATION = 80
CLUSTERS = 9
PCA_DIM = 0

# 基本命令
.PHONY: help run run-full run-pca run-test clean

# 顯示幫助信息
help:
	@echo "Available commands:"
	@echo "  make run         - Run with current settings"
	@echo "  make run-full    - Run with full dimensions (no PCA)"
	@echo "  make run-pca     - Run with PCA reduction"
	@echo "  make run-test    - Run multiple PCA dimensions for testing"
	@echo "  make clean       - Clean output files"
	@echo "\nParameters (can be overridden):"
	@echo "  CORES=$(CORES)"
	@echo "  ITERATIONS=$(ITERATIONS)"
	@echo "  POPULATION=$(POPULATION)"
	@echo "  CLUSTERS=$(CLUSTERS)"
	@echo "  PCA_DIM=$(PCA_DIM)"
	@echo "\nExample:"
	@echo "  make run CLUSTERS=5 POPULATION=100"

# 使用當前設定運行
run:
	$(PYTHON) $(SCRIPT) -c $(CORES) -i $(ITERATIONS) -p $(POPULATION) -k $(CLUSTERS) -d $(PCA_DIM)

# 使用完整維度運行（不降維）
run-full:
	$(PYTHON) $(SCRIPT) -c $(CORES) -i $(ITERATIONS) -p $(POPULATION) -k $(CLUSTERS) -d 0

# 使用PCA降維運行
run-pca:
	$(PYTHON) $(SCRIPT) -c $(CORES) -i $(ITERATIONS) -p $(POPULATION) -k $(CLUSTERS) -d 20

# 運行測試：使用不同的PCA維度
run-test:
	@echo "Testing with different PCA dimensions..."
	@echo "\nTesting with 10 dimensions:"
	$(PYTHON) $(SCRIPT) -c $(CORES) -i $(ITERATIONS) -p $(POPULATION) -k $(CLUSTERS) -d 10
	@echo "\nTesting with 15 dimensions:"
	$(PYTHON) $(SCRIPT) -c $(CORES) -i $(ITERATIONS) -p $(POPULATION) -k $(CLUSTERS) -d 15
	@echo "\nTesting with 20 dimensions:"
	$(PYTHON) $(SCRIPT) -c $(CORES) -i $(ITERATIONS) -p $(POPULATION) -k $(CLUSTERS) -d 20
	@echo "\nTesting with full dimensions:"
	$(PYTHON) $(SCRIPT) -c $(CORES) -i $(ITERATIONS) -p $(POPULATION) -k $(CLUSTERS) -d 0

# 清理輸出文件
clean:
	rm -f res/*.png
	rm -f *.log