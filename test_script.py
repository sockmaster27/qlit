import subprocess

def test_cargo():
    assert subprocess.call(["cargo", "test"]) == 0, "Cargo tests failed"
