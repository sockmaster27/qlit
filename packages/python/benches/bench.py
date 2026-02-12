# To execute locally, run
#
#    uv run pytest packages/python/benches/bench.py --codspeed
#

import random

import pytest
from qlit import (
    CliffordTCircuit,
    simulate_circuit,
    simulate_circuit_gpu,
    simulate_circuit_hybrid,
)

DEFAULT_QUBITS = 100
DEFAULT_GATES = 100
DEFAULT_T_GATES = 5


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
    @pytest.mark.parametrize("qubits", [10, 100, 1000])
    def test_qubits(self, benchmark, qubits, implementation):
        w, circuit = setup(qubits, DEFAULT_GATES, DEFAULT_T_GATES)
        benchmark(implementation, w, circuit)

    @pytest.mark.parametrize("gates", [10, 100, 1000])
    def test_gates(self, benchmark, gates, implementation):
        w, circuit = setup(DEFAULT_QUBITS, gates, DEFAULT_T_GATES)
        benchmark(implementation, w, circuit)

    @pytest.mark.parametrize("t_gates", [0, 5, 10])
    def test_t_gates(self, benchmark, t_gates, implementation):
        w, circuit = setup(DEFAULT_QUBITS, DEFAULT_GATES, t_gates)
        benchmark(implementation, w, circuit)
