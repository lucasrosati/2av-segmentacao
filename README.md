# 2AV EL1-IA 2026.2: Estudo comparativo de segmentação (U-Net + TL: ResNet-34 vs MiT-B0, Kvasir-SEG)

> **Status: concluído.** As 60 execuções (30 seeds × 2 cenários) rodaram no commit `a67a0b3`, numa Tesla T4 do Kaggle. A estatística, as figuras, o paper (`paper/`), o pacote de slides (`results/slides/`) e a demo (`demo/`) saem deste repositório.

## Resultado
- **Decisão sobre H0 (Dice de teste):** não se rejeita H0 pelo teste *t* pareado, a decisão pré-especificada. O MiT-B0 teve Dice médio um pouco maior, mas a diferença não é significativa e o estudo não demonstra superioridade nem equivalência.
- **IoU:** exploratório; não muda a decisão sobre o Dice.
- **Custo:** a ResNet-34 tem mais parâmetros, e o MiT-B0 teve maior latência por imagem.
- **Números** (médias ± dp, IC 95%, *p*-valores, d_z, parâmetros, latência): estão em [`results/slides/key_numbers.md`](results/slides/key_numbers.md) e [`results/tables/stats.csv`](results/tables/stats.csv), gerados por código a partir de `results/runs.csv`. Este README não repete números para não divergir deles.

## Desenho
- A = U-Net, B = Transfer Learning (ImageNet), C = encoder ResNet-34 vs C' = encoder MiT-B0.
- Kvasir-SEG, split fixo 80/10/10 (`splits/`), seeds 0..29 pareados. Teste pareado (t e Wilcoxon), α = 0,05.

## Ambiente
- Python 3.13 e libs fixadas em `pyproject.toml` nas mesmas versões da imagem GPU do Kaggle (resultados locais e do Kaggle saem das mesmas libs).
- Local: `uv sync` (cria `.venv`). `make env` confere se o instalado bate com os pins.
- Kaggle: o kernel fica travado na imagem da v1 (`docker_image_pinning_type: original`, digest `sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f`). torch/torchvision vêm da imagem; instale só `requirements.txt` (smp) e rode `python -m seg2av.env`.

## Uso
```bash
uv sync
make test                            # ruff (check + format) + pytest
make splits                          # M1
make sweep SCENARIO=cnn              # ou SCENARIO=transformer
make collect && make analyze && make slides-assets
make kaggle-push / make kaggle-output
make demo                            # caso de uso: app Gradio em http://127.0.0.1:7860 (ver demo/README.md)
make demo-gif                        # results/slides/demo.gif
make paper                           # paper/main.pdf (IEEEtran, pdflatex + bibtex)
```

- `make demo`, `make demo-gif` e a figura qualitativa precisam dos dados em `data/raw/` (`make data`) e dos `best.pt` das execuções, que não estão no git (ver Avisos).
- `make paper` usa `results/tables/numbers.tex` e `results/figures/`. Sem os dados, a figura qualitativa sai como um aviso em vermelho e o resto compila.

## Avisos
- **Kvasir-SEG:** uso apenas para pesquisa e educação, com citação obrigatória (Jha et al., MMM 2020). Os dados não vão para o git. As figuras com fotos do dataset (`samples.png`, `qualitative.png`, `demo.gif`) e o `paper/main.pdf` também ficam fora do git e são regenerados com `make samples`, `make figures`, `make slides-assets`, `make demo-gif` e `make paper`.
- **Checkpoints** (`best.pt`) não são versionados. Cada execução guarda só config, métricas e históricos em `results/runs/<cenário>_seed<n>/`.
- `kaggle.json` **nunca** é comitado.
- Nenhum número no paper ou nos slides pode ser digitado à mão: tudo sai de `results/`.
