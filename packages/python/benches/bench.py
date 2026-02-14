# To execute locally, run
#
#    uv run pytest packages/python/benches/bench.py --codspeed
#

import random
import os

import pytest
from qlit import (
    CliffordTCircuit,
    simulate_circuit,
    simulate_circuit_gpu,
    simulate_circuit_hybrid,
)


def setup(qubits, gates, t_gates):
    seed = 123
    random.seed(seed)
    w = format(random.getrandbits(qubits), f"00{qubits}b")
    circuit = CliffordTCircuit.random(qubits, gates, t_gates, seed)
    return w, circuit


@pytest.mark.parametrize(
    "implementation", [simulate_circuit, simulate_circuit_gpu, simulate_circuit_hybrid]
)
class TestPython:
    def test_small(self, benchmark, implementation):
        w, circuit = setup(8, 64, 5)
        benchmark(implementation, w, circuit)


    @pytest.mark.skipif(condition="CI" in os.environ, reason="Too slow for CI")
    def test_large(self, benchmark, implementation):
        w, circuit = setup(32, 512, 17)
        benchmark(implementation, w, circuit)
