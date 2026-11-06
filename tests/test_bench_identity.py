"""Hermetic tests for scripts/bench_identity.py (no GPU, no ROCm, no model)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bench_identity  # noqa: E402


def _make_venv(root: Path, payload: bytes) -> Path:
    lib = root / "lib" / "python3.14" / "site-packages" / "mlx" / "lib"
    lib.mkdir(parents=True)
    (lib / "libmlx.so").write_bytes(payload)
    return root


def _make_models(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model-00001.safetensors").write_bytes(b"weights")
    (root / "config.json").write_text("{}")
    (root / "notes.md").write_bytes(b"not a model file")
    return root


def test_block_digests_wheel_and_model_files(tmp_path: Path) -> None:
    venv = _make_venv(tmp_path / "venv", b"wheel-bytes")
    models = _make_models(tmp_path / "models")
    bench = tmp_path / "bench.py"
    bench.write_text("print('bench')\n")

    lines, missing = bench_identity.build_block(venv, models, [bench])
    text = "\n".join(lines)

    assert missing == []
    assert hashlib.sha256(b"wheel-bytes").hexdigest() in text
    assert hashlib.sha256(b"weights").hexdigest() in text
    # non-model files are not digested as model weights
    assert "notes.md" not in text
    assert "identity_digest:" in text


def test_missing_inputs_are_reported_not_silently_dropped(tmp_path: Path) -> None:
    lines, missing = bench_identity.build_block(
        tmp_path / "absent-venv", tmp_path / "absent-models", [tmp_path / "absent-bench.py"]
    )
    text = "\n".join(lines)

    assert "libmlx" in missing
    assert "models" in missing
    assert any(m.startswith("bench:") for m in missing)
    assert "MISSING" in text


def test_env_block_never_leaks_unrelated_variables(monkeypatch) -> None:
    monkeypatch.setenv("MTP6_TAPE", "hybrid")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "should-not-appear")

    lines = bench_identity.env_block()
    text = "\n".join(lines)

    assert "MTP6_TAPE=hybrid" in text
    assert "should-not-appear" not in text


def test_digest_is_stable_for_identical_content(tmp_path: Path) -> None:
    first = bench_identity.digest_body(["a: 1", "b: 2"])
    second = bench_identity.digest_body(["a: 1", "b: 2"])
    third = bench_identity.digest_body(["a: 1", "b: 3"])

    assert first == second
    assert first != third
    assert len(first) == 16


def test_main_writes_block_to_out_path(tmp_path: Path) -> None:
    venv = _make_venv(tmp_path / "venv", b"x")
    models = _make_models(tmp_path / "models")
    out = tmp_path / "nested" / "identity.txt"

    code = bench_identity.main(["--venv", str(venv), "--models", str(models), "--out", str(out)])

    assert code == 0
    assert out.is_file()
    assert "identity_digest:" in out.read_text()


def test_host_is_redacted_by_default_and_opt_in_when_asked() -> None:
    default_lines, _ = bench_identity.build_block(None, None, [])
    opt_in_lines, _ = bench_identity.build_block(None, None, [], include_host=True)

    assert any(line.startswith("host: redacted") for line in default_lines)
    assert not any("host: redacted" in line for line in opt_in_lines)
    assert any(line.startswith("host: ") for line in opt_in_lines)


def test_models_not_requested_is_marked_not_silently_absent() -> None:
    lines, missing = bench_identity.build_block(None, None, [])
    text = "\n".join(lines)

    assert "models" in missing
    assert "model: NOT REQUESTED" in text


def test_forgotten_venv_never_pins_a_system_wheel() -> None:
    """A missing --venv must be a MISSING field, never a confidently pinned wrong wheel."""
    lines, missing = bench_identity.build_block(None, None, [])
    text = "\n".join(lines)

    assert "libmlx" in missing
    assert "libmlx: MISSING" in text
    assert "libmlx: path=" not in text


def test_missing_inputs_make_exit_code_two() -> None:
    assert bench_identity.main([]) == 2
