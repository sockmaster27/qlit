import subprocess

import pytest


# TODO: Make this marking more granular. Many tests in here do not depend on GPU
@pytest.mark.gpu
def test_cargo():
    assert subprocess.call(["cargo", "test"]) == 0, "Cargo tests failed"
