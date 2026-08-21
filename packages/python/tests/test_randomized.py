import json
import random
from pathlib import Path

import pytest
from qlit import (
    CliffordTCircuit,
    simulate_circuit,
    simulate_circuit_gpu,
    simulate_circuit_hybrid,
)

FIXTURES_PATH = Path(__file__).parent / "randomized_fixtures.json"


class TestQlitRandomized:
    def setup_test_against_statevector(self):
        with open(FIXTURES_PATH, "r") as f:
            fixtures = json.load(f)
        for fixture in fixtures:
            circuit = CliffordTCircuit.random(
                fixture["qubits"], 
                fixture["gates"], 
                fixture["t_gates"], 
                fixture["seed"],
            )
            for w_data in fixture["w_data"]:
                w = w_data["w"]
                w_coeff_qiskit = complex(w_data["w_coeff_re"], w_data["w_coeff_im"])
                yield w, circuit, w_coeff_qiskit
    
    def test_against_statevector(self):
        for w, circuit, w_coeff_qiskit in self.setup_test_against_statevector():
            assert simulate_circuit(w, circuit) == pytest.approx(w_coeff_qiskit), f"\nCPU: {w=} \n{circuit.gates=}"
            
    @pytest.mark.gpu
    def test_against_statevector_gpu(self):
        for w, circuit, w_coeff_qiskit in self.setup_test_against_statevector():
            assert simulate_circuit_gpu(w, circuit) == pytest.approx(w_coeff_qiskit), f"\nGPU: {w=} \n{circuit.gates=}"

    @pytest.mark.gpu
    def test_against_statevector_hybrid(self):
        for w, circuit, w_coeff_qiskit in self.setup_test_against_statevector():
            assert simulate_circuit_hybrid(w, circuit) == pytest.approx(w_coeff_qiskit), f"\nHybrid: {w=} \n{circuit.gates=}"

    @pytest.mark.gpu
    def test_implementations_against_eachother(self):
        iterations = 5
        w_samples = 5
        for i in range(iterations):
            qubits = 130
            t_gates = 3
            gates = 1000
            seed = 1234 + i
            circuit = CliffordTCircuit.random(qubits, gates, t_gates, seed)

            for _ in range(w_samples):
                w = format(random.getrandbits(qubits), f"00{qubits}b")
                simulate_circuit_res = simulate_circuit(w, circuit)
                simulate_circuit_gpu_res = simulate_circuit_gpu(w, circuit)
                simulate_circuit_hybrid_res = simulate_circuit_hybrid(w, circuit)

                assert simulate_circuit_res == pytest.approx(simulate_circuit_gpu_res), f"\nCPU/GPU: {w=} \n{circuit.gates=}"
                assert simulate_circuit_gpu_res == pytest.approx(simulate_circuit_hybrid_res), f"\nGPU/Hybrid: {w=} \n{circuit.gates=}"
