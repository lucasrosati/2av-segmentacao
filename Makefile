# Alvos do projeto (M0 a M9): dados, treino, análise, slides, demo e paper.
PY ?= uv run python
SCENARIO ?= cnn
CONFIG_cnn := configs/cnn_resnet34.yaml
CONFIG_transformer := configs/transformer_mitb0.yaml
KERNEL_DIR := kaggle

.PHONY: setup env test lint data splits samples overfit baseline sweep pilot collect analyze stats figures slides-assets demo demo-gif paper kaggle-push kaggle-status kaggle-output

setup:          ## instala dependências
	uv sync
env:            ## confere libs instaladas x pins do pyproject (rodar também no Kaggle)
	$(PY) -m seg2av.env
lint:
	uv run ruff check . && uv run ruff format --check .
test: lint      ## M0: lint + testes unitários e smoke
	uv run pytest -q
data:           ## M1: baixa e verifica o Kvasir-SEG (local)
	bash scripts/download_data.sh
splits:         ## M1/D2: gera splits/split_v1.json + .sha256 (determinístico)
	$(PY) scripts/make_splits.py --seed 0 --out splits/split_v1.json
samples:        ## M1: results/figures/samples.png
	$(PY) -m seg2av.plots --target samples
overfit:        ## M2: sanidade, 8 imagens até Dice > 0,9
	$(PY) scripts/overfit.py --config $(CONFIG_$(SCENARIO))
baseline:       ## M3: piso de sanidade (máscara zero) no teste
	$(PY) -m seg2av.evaluate
pilot:          ## M4: 5 seeds por cenário
	$(PY) scripts/run_sweep.py --config $(CONFIG_$(SCENARIO)) --seeds 0-4 --resume
sweep:          ## M5: 30 seeds (retomável)
	$(PY) scripts/run_sweep.py --config $(CONFIG_$(SCENARIO)) --seeds 0-29 --resume
collect:        ## D3: consolida results/runs/*/metrics.json -> results/runs.csv
	$(PY) scripts/collect.py
stats:          ## M6
	$(PY) -m seg2av.stats
figures:        ## M7
	$(PY) -m seg2av.plots --target paper
analyze: collect stats figures
slides-assets:  ## M7b: results/slides/ (PNG 1920x1080, tabelas, key_numbers.md, HANDOFF.md)
	$(PY) -m seg2av.slides
demo:           ## M8: app Gradio local (http://127.0.0.1:7860)
	uv run --extra demo python demo/app.py
demo-gif:       ## M8: results/slides/demo.gif a partir das runs medianas
	$(PY) -m seg2av.demo
paper:          ## M9
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex && bibtex main && pdflatex -interaction=nonstopmode -halt-on-error main.tex && pdflatex -interaction=nonstopmode -halt-on-error main.tex
kaggle-push:    ## envia o notebook (GPU habilitada em kaggle/kernel-metadata.json)
	uv run kaggle kernels push -p $(KERNEL_DIR)
kaggle-status:
	uv run kaggle kernels status $$($(PY) -c "import json;print(json.load(open('$(KERNEL_DIR)/kernel-metadata.json'))['id'])")
kaggle-output:  ## baixa saídas para results_kaggle/
	uv run kaggle kernels output $$($(PY) -c "import json;print(json.load(open('$(KERNEL_DIR)/kernel-metadata.json'))['id'])") -p results_kaggle/
