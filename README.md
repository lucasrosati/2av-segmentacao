# 2AV EL1-IA 2026.2: Estudo comparativo de segmentação (U-Net + TL: ResNet-34 vs MiT-B0, Kvasir-SEG)

> **Status: esqueleto (scaffold).** Nada aqui está implementado ainda e não há resultados. Os arquivos contêm stubs e TODOs.

- Plano: documento local (não versionado).

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
```

## Avisos
- Kvasir-SEG: uso apenas para pesquisa/educação, com citação obrigatória (Jha et al., 2020). Os dados não vão para o git.
- `kaggle.json` **nunca** é comitado.
- Nenhum número no paper ou nos slides pode ser digitado à mão: tudo sai de `results/`.
