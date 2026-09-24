"""Read-only syntax, config, checksum, and distribution-content checks.

Requires Python and PyYAML, not PyTorch, datasets, or network access.
This is a packaging check, not a complete credential or license audit.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".git", ".venv", "venv", "build", "dist"}
PRIVATE_ROOTS = {"data", "manifests", "weights", "checkpoints", "runs", "logs", "private"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".npy", ".npz", ".png", ".jpg", ".jpeg", ".log", ".pem", ".key"}


def package_files() -> list[Path]:
    return sorted(p for p in ROOT.rglob("*") if p.is_file()
                  and not any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in p.relative_to(ROOT).parts)
                  and p.relative_to(ROOT).parts[0] not in PRIVATE_ROOTS)


def merge(left: dict, right: dict) -> dict:
    result = copy.deepcopy(left)
    for key, value in right.items():
        result[key] = merge(result[key], value) if isinstance(result.get(key), dict) and isinstance(value, dict) else value
    return result


def load_config(path: Path, ancestors: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in ancestors or ROOT / "configs" not in path.parents:
        raise ValueError(f"Invalid config inheritance: {path.name}")
    cfg = yaml.safe_load(path.read_text())
    parent = cfg.pop("extends", None)
    return merge(load_config(path.parent / parent, (*ancestors, path)), cfg) if parent else cfg


def differences(left: dict, right: dict, prefix: str = "") -> set[str]:
    result = set()
    for key in left.keys() | right.keys():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(left.get(key), dict) and isinstance(right.get(key), dict):
            result |= differences(left[key], right[key], name)
        elif left.get(key) != right.get(key):
            result.add(name)
    return result


def main() -> None:
    files = package_files()
    for path in files:
        assert not path.is_symlink(), f"Symlink in release: {path.relative_to(ROOT)}"
        assert path.suffix.lower() not in FORBIDDEN_SUFFIXES, f"Private/binary artifact: {path.name}"
        assert not path.name.startswith(".env"), f"Environment file: {path.name}"
        assert path.stat().st_size < 2_000_000, f"Unexpected large file: {path.name}"
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----", text), path.name
        if path.suffix == ".py":
            ast.parse(text, filename=str(path))
    configs = {path.stem: load_config(path) for path in (ROOT / "configs").glob("*.yaml")}
    for name, cfg in configs.items():
        assert cfg["train"]["epochs"] == 120, name
        assert cfg["train"]["selection_metric"] == "val_seg_dice", name
        assert cfg["model"]["image_size"] == 768, name
        assert cfg["data"]["source_domain"] == "REFUGE", name
        for value in (cfg["data"]["root"], cfg["data"]["train_manifest"], cfg["data"]["val_manifest"], cfg["model"]["checkpoint"]):
            assert not Path(value).is_absolute(), (name, "nonportable path")
    full = configs["cupgeo"]
    assert differences(full, configs["cupgeo_no_sg"]) == {"model.detach_cup_from_od"}
    assert configs["cupgeo_no_sg"]["model"]["detach_cup_from_od"] is False
    assert differences(full, configs["cupgeo_no_vra"]) == {
        "model.vertical_geometry_head", "model.vertical_geometry_calibration",
        "train.vertical_rim_allocation.enabled", "train.vertical_rim_allocation.weight"}
    for name in ("cupgeo", "cupgeo_no_ratio", "cupgeo_no_vra", "cp_baseline"):
        assert configs[name]["model"]["detach_cup_from_od"] is True, name
    assert configs["cupgeo_no_ratio"]["train"]["loss"].get("vcdr_weight", 0) == 0
    assert configs["cp_baseline"]["train"]["loss"].get("vcdr_weight", 0) == 0
    checksums = {}
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert name not in checksums, f"Duplicate checksum entry: {name}"
        checksums[name] = digest
    actual = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in files if p.name != "SHA256SUMS"}
    assert actual == checksums, "File inventory/checksums differ from prepared release"
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    for item in manifest["copied_files"]:
        assert actual[item["file"]] == item["source_sha256"], item["file"]
    print(f"PASS: {sum(p.suffix == '.py' for p in files)} Python files parse; {len(configs)} configs resolve")
    print(f"PASS: stage-2 ablation switches and relative data/weight paths; {len(checksums)} checksums")
    print(f"PASS: {len(manifest['copied_files'])} copied files match their recorded source hashes")
    print("Checked tracked source content only; ignored local runs/data are not release artifacts.")
    print("This does not validate full training, numerical reproduction, credentials, or licensing.")


if __name__ == "__main__":
    main()
