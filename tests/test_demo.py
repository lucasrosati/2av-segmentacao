import numpy as np
from PIL import Image
from test_plots import MODEL, NAMES, _fake_checkpoint, _fake_dataset, _fake_results

from seg2av.demo import demo_gif, load_models, overlay, segment
from seg2av.plots import median_seed
from seg2av.stats import ALT, BASE


def test_overlay_only_touches_mask():
    image = np.full((4, 4, 3), 100, np.uint8)
    mask = np.zeros((4, 4), np.uint8)
    mask[0, 0] = 1
    out = overlay(image, mask)
    assert out[0, 0, 0] > 100 and (out[1:] == 100).all()


def test_segment_and_gif_with_median_checkpoints(tmp_path):
    rng = np.random.default_rng(0)
    runs = _fake_results(tmp_path, rng)
    _fake_dataset(tmp_path / "data", rng)
    for s in (BASE, ALT):
        _fake_checkpoint(tmp_path / "runs" / f"{s}_seed{median_seed(runs, s)}", MODEL[s])
    models = load_models(tmp_path)
    assert {s: m["seed"] for s, m in models.items()} == {s: median_seed(runs, s) for s in models}
    image = rng.integers(0, 255, (80, 96, 3), np.uint8)
    out = segment(models, image)
    assert all(o.shape == image.shape and 0 <= frac <= 1 for o, frac in out.values())
    gif = tmp_path / "demo.gif"
    demo_gif(models, tmp_path / "data", NAMES[:2], gif)
    with Image.open(gif) as im:
        assert im.size == (1920, 1080) and im.n_frames == 2
