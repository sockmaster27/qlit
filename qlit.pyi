from dataclasses import dataclass
from typing import Iterable


class CliffordGate:
    @dataclass
    class H:
        a: int

    @dataclass
    class S:
        a: int

    @dataclass
    class Cnot:
        a: int
        b: int


class CliffordCircuit:
    def __init__(self, qubits: int, gates: Iterable[CliffordGate]) -> None: ...
    @classmethod
    def random(qubits: int, gates: int, seed: int) -> CliffordCircuit: ...

    @property
    def qubits(self) -> int: ...
    @property
    def gates(self) -> list[CliffordGate]: ...


class BasisStateProbability:
    One: BasisStateProbability
    InBetween: BasisStateProbability
    Zero: BasisStateProbability


def simulate_clifford_circuit_gpu(w: str, circuit: CliffordCircuit) -> BasisStateProbability: ...
def coeff_ratio(w1: str, w2: str, circuit: CliffordCircuit) -> BasisStateProbability: ...
def clifford_phase(w: str, circuit: CliffordCircuit) -> BasisStateProbability: ...
