from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Any

from .config_guard import ConfigGuard, DEFAULT_STATE_DIR, CONFIG_FILENAME
from .pusula_types import PusulaConfig

logger = logging.getLogger("ceviz.pusula.snapshot")


def get_snapshots_dir(state_dir: Path | str | None = None) -> Path:
    base = Path(state_dir) if state_dir else DEFAULT_STATE_DIR
    snap_dir = base / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    return snap_dir


def list_snapshots(state_dir: Path | str | None = None) -> list[dict[str, Any]]:
    snap_dir = get_snapshots_dir(state_dir)
    snapshots: list[dict[str, Any]] = []

    for f in sorted(snap_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            meta = data.get("_snapshot_meta", {})
            snapshots.append({
                "name": f.stem,
                "path": str(f),
                "created_at": meta.get("created_at") or datetime.datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                "description": meta.get("description", ""),
                "routing_mode": data.get("routing_mode", "single_turn"),
                "enabled": data.get("enabled", True),
                "default_group": data.get("default_group", ""),
                "groups": list(data.get("groups", {}).keys()),
            })
        except Exception as exc:
            snapshots.append({
                "name": f.stem,
                "path": str(f),
                "error": str(exc),
            })

    return snapshots


def create_snapshot(
    name: str,
    description: str = "",
    state_dir: Path | str | None = None,
) -> Path:
    base = Path(state_dir) if state_dir else DEFAULT_STATE_DIR
    current_config_path = base / CONFIG_FILENAME
    if not current_config_path.is_file():
        raise FileNotFoundError(f"Cannot snapshot missing configuration: {current_config_path}")

    snap_dir = get_snapshots_dir(base)
    clean_name = name[:-5] if name.endswith(".json") else name
    target_path = snap_dir / f"{clean_name}.json"

    data = json.loads(current_config_path.read_text(encoding="utf-8"))
    data["_snapshot_meta"] = {
        "name": clean_name,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "description": description,
    }

    tmp_path = target_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(target_path)
    logger.info(f"[pusula.snapshot] Created snapshot '{clean_name}' at {target_path}")
    return target_path


def restore_snapshot(name: str, state_dir: Path | str | None = None) -> PusulaConfig:
    base = Path(state_dir) if state_dir else DEFAULT_STATE_DIR
    snap_dir = get_snapshots_dir(base)
    clean_name = name[:-5] if name.endswith(".json") else name
    target_path = snap_dir / f"{clean_name}.json"

    if not target_path.is_file():
        raise FileNotFoundError(f"Snapshot '{name}' not found at {target_path}")

    data = json.loads(target_path.read_text(encoding="utf-8"))
    # Remove snapshot metadata before saving as active config
    config_dict = {k: v for k, v in data.items() if not k.startswith("_")}
    config = PusulaConfig.from_dict(config_dict)

    guard = ConfigGuard(state_dir=base)
    guard.save_config(config)
    logger.info(f"[pusula.snapshot] Restored snapshot '{clean_name}' to {guard.config_path}")
    return config


def set_routing_mode(mode: str, state_dir: Path | str | None = None) -> PusulaConfig:
    valid_modes = {"context_aware", "single_turn", "disabled"}
    clean_mode = mode.strip().lower()
    if clean_mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Choose from: {sorted(valid_modes)}")

    guard = ConfigGuard(state_dir=state_dir)
    config, _ = guard.load_config()

    if clean_mode == "disabled":
        config.enabled = False
        config.routing_mode = "disabled"
    else:
        config.enabled = True
        config.routing_mode = clean_mode

    guard.save_config(config)
    logger.info(f"[pusula.snapshot] Updated routing mode to: '{clean_mode}'")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="cevizPusula Snapshot & Mode Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    subparsers.add_parser("list", help="List all saved snapshots")

    # create
    create_parser = subparsers.add_parser("create", help="Create a snapshot of current config")
    create_parser.add_argument("name", help="Snapshot name (e.g. v1-baseline, before-test)")
    create_parser.add_argument("--description", "-d", default="", help="Optional description")

    # restore
    restore_parser = subparsers.add_parser("restore", help="Restore configuration from a snapshot")
    restore_parser.add_argument("name", help="Snapshot name to restore")

    # mode
    mode_parser = subparsers.add_parser("mode", help="View or change current routing mode")
    mode_parser.add_argument("mode", nargs="?", choices=["context_aware", "single_turn", "disabled"], help="New mode")

    args = parser.parse_args()

    if args.command == "list":
        snaps = list_snapshots()
        if not snaps:
            print("No snapshots found.")
            return
        print(f"{'NAME':<20} {'MODE':<15} {'CREATED':<25} {'DESCRIPTION'}")
        print("-" * 75)
        for s in snaps:
            if "error" in s:
                print(f"{s['name']:<20} ERROR: {s['error']}")
            else:
                print(f"{s['name']:<20} {s['routing_mode']:<15} {s['created_at'][:19]:<25} {s['description']}")

    elif args.command == "create":
        p = create_snapshot(args.name, description=args.description)
        print(f"Snapshot created successfully: {p.name}")

    elif args.command == "restore":
        cfg = restore_snapshot(args.name)
        print(f"Restored snapshot '{args.name}'. Mode: {cfg.routing_mode}, Enabled: {cfg.enabled}")

    elif args.command == "mode":
        if args.mode:
            cfg = set_routing_mode(args.mode)
            print(f"Routing mode updated to: {cfg.routing_mode} (enabled: {cfg.enabled})")
        else:
            guard = ConfigGuard()
            cfg, _ = guard.load_config()
            print(f"Current routing mode: {cfg.routing_mode} (enabled: {cfg.enabled})")


if __name__ == "__main__":
    main()
