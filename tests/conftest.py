import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vne_gen.problem.generators import generate_workload, load_geant  # noqa: E402


@pytest.fixture(scope="session")
def geant():
    return load_geant(np.random.default_rng(0))


@pytest.fixture(scope="session")
def workload():
    return generate_workload(np.random.default_rng(1), 200, 0.06)
