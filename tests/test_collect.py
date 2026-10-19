import json
import runpy

from seg2av.env import PYPROJECT

collect = runpy.run_path(str(PYPROJECT.parent / "scripts/collect.py"))["collect"]


def test_collect_ignora_run_sem_done_e_ordena(tmp_path):
    for name, seed, done in [("b_seed1", 1, True), ("a_seed0", 0, True), ("a_seed1", 1, False)]:
        d = tmp_path / name
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps({"scenario": name[0], "seed": seed}))
        if done:
            (d / "DONE").touch()
    df = collect(tmp_path)
    assert df[["scenario", "seed"]].values.tolist() == [["a", 0], ["b", 1]]
