"""M8: demo Gradio. Envie uma imagem de colonoscopia; o app mostra a máscara prevista por cada
encoder (run de Dice mediano de cada cenário) sobreposta em vermelho. Rodar com `make demo`."""

import gradio as gr

from seg2av.data import resolve_root
from seg2av.demo import load_models, segment, test_names
from seg2av.plots import LABEL, SCENARIOS

MODELS = load_models()


def run(image):
    try:
        out = segment(MODELS, image)
    except ValueError as e:
        raise gr.Error(f"Não foi possível segmentar: {e}.") from e
    return [out[s][0] for s in SCENARIOS] + [
        "\n".join(f"{LABEL[s]}: {100 * out[s][1]:.1f}% da imagem marcada" for s in SCENARIOS)
    ]


with gr.Blocks(title="Segmentação de pólipos: ResNet-34 × MiT-B0") as app:
    gr.Markdown(
        "# Segmentação de pólipos (Kvasir-SEG)\n"
        "U-Net com *encoder* ResNet-34 (CNN) e MiT-B0 (Transformer), run de Dice mediano de cada "
        f"cenário (seeds {', '.join(str(MODELS[s]['seed']) for s in SCENARIOS)})."
    )
    with gr.Row():
        inp = gr.Image(type="numpy", label="Imagem de colonoscopia")
        outs = [gr.Image(label=LABEL[s]) for s in SCENARIOS]
    info = gr.Textbox(label="Área prevista")
    gr.Button("Segmentar").click(run, inp, [*outs, info])
    root = resolve_root()
    if root.exists():
        gr.Examples([[str(root / "images" / f"{n}.jpg")] for n in test_names(4)], inp)

if __name__ == "__main__":
    app.launch()
