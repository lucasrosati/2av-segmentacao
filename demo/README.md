# demo/

Caso de uso do projeto (M8): um app Gradio que segmenta pólipos com os dois encoders lado a lado.

## O que faz
- Recebe uma imagem de colonoscopia.
- Mostra a máscara prevista por cada cenário (U-Net + ResNet-34 e U-Net + MiT-B0), sobreposta em vermelho, e a fração da imagem marcada como pólipo.
- Usa, de cada cenário, o `best.pt` (checkpoint de melhor validação) da execução cujo Dice de teste é o mais próximo da mediana do cenário (`seg2av.plots.median_seed`). Essa escolha serve só para a demonstração. A análise estatística usa as 30 execuções de cada cenário.
- Oferece como exemplos quatro imagens do conjunto de teste, igualmente espaçadas no split, quando os dados estão em `data/raw/`.

É um caso de uso educacional, sem validação clínica.

## Como rodar
```bash
uv sync --extra demo   # instala o Gradio (extra "demo" do pyproject.toml)
make demo              # http://127.0.0.1:7860
make demo-gif          # results/slides/demo.gif: 5 imagens do teste, máscara real × ResNet-34 × MiT-B0
```

Pré-requisitos, que não estão no git:
- `results/runs/<cenário>_seed<n>/best.pt` das execuções medianas, gerados pelo sweep;
- `data/raw/kvasir-seg/` (`make data`) para os exemplos e o GIF.

## Código
- `demo/app.py`: interface Gradio.
- `src/seg2av/demo.py`: carregamento dos modelos, inferência, sobreposição e GIF, sem depender do Gradio. Testado em `tests/test_demo.py`.
