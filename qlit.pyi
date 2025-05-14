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


class CliffordTGate:
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

    @dataclass
    class T:
        a: int


class CliffordCircuit:
    def __init__(self, qubits: int, gates: Iterable[CliffordGate]) -> None: ...
    @classmethod
    def random(qubits: int, gates: int, seed: int) -> CliffordCircuit: ...

    @property
    def qubits(self) -> int: ...
    @property
    def gates(self) -> list[CliffordGate]: ...


class CliffordTCircuit:
    def __init__(self, qubits: int, gates: Iterable[CliffordTGate]) -> None: ...
    @classmethod
    def random(qubits: int, gates: int, t_gates: int, seed: int) -> CliffordTCircuit: ...

    @property
    def qubits(self) -> int: ...
    @property
    def t_gates(self) -> int: ...
    @property
    def gates(self) -> list[CliffordTGate]: ...


class BasisStateProbability:
    One: BasisStateProbability
    InBetween: BasisStateProbability
    Zero: BasisStateProbability


def simulate_clifford_circuit_gpu(w: str, circuit: CliffordCircuit) -> BasisStateProbability: ...
def coeff_ratio(w1: str, w2: str, circuit: CliffordCircuit) -> complex: ...
def clifford_phase(w: str, circuit: CliffordCircuit) -> complex: ...
def simulate_circuit(w: str, circuit: CliffordTCircuit) -> complex: ...
