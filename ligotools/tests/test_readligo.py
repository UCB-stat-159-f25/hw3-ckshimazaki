import os
import numpy as np
import pytest
from ligotools import readligo as rl


def test_loaddata_basic():
    fname = os.path.join("data", "H-H1_LOSC_4_V2-1126259446-32.hdf5")
    strain, time, chan = rl.loaddata(fname, "H1")
    assert isinstance(strain, np.ndarray)
    assert isinstance(time, np.ndarray)
    assert isinstance(chan, dict)
    assert len(strain) == len(time)
    dt = np.mean(np.diff(time))
    assert np.isclose(dt, 1 / 4096, atol=1e-5)


def test_read_frame_error():
    with pytest.raises((ImportError, ModuleNotFoundError, TypeError)):
        rl.read_frame("fake_file.gwf", ifo=None)
