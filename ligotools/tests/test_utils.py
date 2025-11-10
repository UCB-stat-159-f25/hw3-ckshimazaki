import numpy as np
import os
from ligotools import utils

def test_whiten():
    t = np.linspace(0, 1, 2048)
    strain = np.sin(2 * np.pi * 120 * t) + 0.3 * np.random.randn(2048)
    dt = 1.0 / 2048
    interp_psd = lambda f: np.ones_like(f)
    w = utils.whiten(strain, interp_psd, dt)
    assert isinstance(w, np.ndarray)
    assert w.shape == strain.shape
    assert np.isfinite(w).all()
    assert np.std(w) != 0
    assert not np.allclose(w, strain)

def test_write_wavfile(tmp_path):
    fs = 16000
    t = np.linspace(0, 1, fs, endpoint=False)
    data = 0.8 * np.sin(2 * np.pi * 440 * t)
    out = tmp_path / "sound.wav"
    utils.write_wavfile(out, fs, data)
    assert out.exists()
    assert os.path.getsize(out) > 1000
    assert str(out).endswith(".wav")
