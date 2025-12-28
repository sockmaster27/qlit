from dataclasses import dataclass
from typing import Iterable


class CliffordTGate:
    @dataclass
    class X:
        a: int
    @dataclass
    class Y:
        a: int
    @dataclass
    class Z:
        a: int

    @dataclass
    class H:
        a: int

    @dataclass
    class S:
        a: int
    @dataclass
    class Sdg:
        a: int

    @dataclass
    class Cnot:
        a: int
        b: int

    @dataclass
    class Cz:
        a: int
        b: int

    @dataclass
    class T:
        a: int
    @dataclass
    class Tdg:
        a: int


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


def simulate_circuit(w: str, circuit: CliffordTCircuit) -> complex: ...
