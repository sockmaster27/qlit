# To execute locally, run
#
#    uv run pytest packages/python/benches/bench.py --codspeed
#

import pytest
import random
from qlit import CliffordTCircuit, simulate_circuit, simulate_circuit_parallel1, simulate_circuit_parallel, simulate_circuit_parallel2

target = simulate_circuit_parallel2

def setup(qubits, gates, t_gates):
    seed = 123
    random.seed(seed)
    w = format(random.getrandbits(qubits), f"00{qubits}b")
    circuit = CliffordTCircuit.random(qubits, gates, t_gates, seed)
    return w, circuit

@pytest.mark.parametrize("qubits", [10, 100, 1000])
def test_qubits(benchmark, qubits):
    gates = 1000
    t_gates = 5
    w, circuit = setup(qubits, gates, t_gates)
    benchmark(target, w, circuit)

@pytest.mark.parametrize("gates", [100, 1000, 10_000])
def test_gates(benchmark, gates):
    qubits = 100
    t_gates = 5
    w, circuit = setup(qubits, gates, t_gates)
    benchmark(target, w, circuit)

@pytest.mark.parametrize("t_gates", [0, 5, 10])
def test_t_gates(benchmark, t_gates):
    qubits = 100
    gates = 1000
    w, circuit = setup(qubits, gates, t_gates)
    benchmark(target, w, circuit)
