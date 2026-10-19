from seg2av.env import mismatches, pins


def test_pins_le_pyproject():
    p = pins()
    assert p["torch"] == "2.11.0"
    assert p["segmentation-models-pytorch"] == "0.5.0"


def test_mismatches_ignora_sufixo_local_e_acusa_divergencia():
    got = {"torch": "2.11.0+cu128", "pandas": "3.0.6", "numpy": None}.get
    out = mismatches({"torch": "2.11.0", "pandas": "2.3.3", "numpy": "2.1.3"}, get=got)
    assert out == ["pandas: pin 2.3.3, instalado 3.0.6", "numpy: pin 2.1.3, instalado None"]


def test_ambiente_local_bate_com_pins():
    assert mismatches(pins()) == []
