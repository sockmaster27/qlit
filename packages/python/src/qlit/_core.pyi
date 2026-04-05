from dataclasses import dataclass
from typing import Iterable


class CliffordTGate:
    @dataclass
    class X(CliffordTGate):
        qubit: int
    @dataclass
    class Y(CliffordTGate):
        qubit: int
    @dataclass
    class Z(CliffordTGate):
        qubit: int
    @dataclass
    class S(CliffordTGate):
        qubit: int
    @dataclass
    class Sdg(CliffordTGate):
        qubit: int
    @dataclass
    class H(CliffordTGate):
        qubit: int
    @dataclass
    class T(CliffordTGate):
        qubit: int
    @dataclass
    class Tdg(CliffordTGate):
        qubit: int
    @dataclass
    class Cnot(CliffordTGate):
        control: int
        target: int
    @dataclass
    class Cz(CliffordTGate):
        control: int
        target: int

class CliffordTCircuit:
    def __init__(self, qubits: int, gates: Iterable[CliffordTGate]) -> None: ...
    @staticmethod
    def random(qubits: int, gates: int, t_gates: int, seed: int) -> CliffordTCircuit: ...

    @property
    def qubits(self) -> int: ...
    @property
    def t_gates(self) -> int: ...
    @property
    def gates(self) -> list[CliffordTGate]: ...


def simulate_circuit(w: str, circuit: CliffordTCircuit) -> complex: ...
def simulate_circuit_gpu(w: str, circuit: CliffordTCircuit) -> complex: ...
def simulate_circuit_hybrid(w: str, circuit: CliffordTCircuit) -> complex: ...
