from collections.abc import Iterable
from dataclasses import dataclass

class Gate:
    @dataclass
    class X(Gate):
        qubit: int
    @dataclass
    class Y(Gate):
        qubit: int
    @dataclass
    class Z(Gate):
        qubit: int
    @dataclass
    class S(Gate):
        qubit: int
    @dataclass
    class Sdg(Gate):
        qubit: int
    @dataclass
    class H(Gate):
        qubit: int
    @dataclass
    class T(Gate):
        qubit: int
    @dataclass
    class Tdg(Gate):
        qubit: int
    @dataclass
    class Rx(Gate):
        angle: float
        qubit: int
    @dataclass
    class Ry(Gate):
        angle: float
        qubit: int
    @dataclass
    class Rz(Gate):
        angle: float
        qubit: int
    @dataclass
    class Cnot(Gate):
        control: int
        target: int
    @dataclass
    class Cz(Gate):
        control: int
        target: int

class Circuit:
    def __init__(self, qubits: int, gates: Iterable[Gate]) -> None: ...
    @staticmethod
    def random_clifford_t(qubits: int, gates: int, t_gates: int, seed: int) -> Circuit: ...

    @property
    def qubits(self) -> int: ...
    @property
    def gates(self) -> list[Gate]: ...


def simulate_circuit(w: str, circuit: Circuit) -> complex: ...
def simulate_circuit_gpu(w: str, circuit: Circuit) -> complex: ...
def simulate_circuit_hybrid(w: str, circuit: Circuit) -> complex: ...
