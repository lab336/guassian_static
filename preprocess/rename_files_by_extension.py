"""Rename files in a folder to 1.ext, 2.ext ... per extension.

Examples:
    python preprocess/rename_files_by_extension.py data/images
    python preprocess/rename_files_by_extension.py data/images --execute
    python preprocess/rename_files_by_extension.py data/images --extensions .tif .png .jpg --execute
"""

from __future__ import annotations

import argparse
import re
import uuid
from collections import defaultdict
from pathlib import Path


def natural_key(path: Path) -> list[object]:
    """Sort names like 2.jpg before 10.jpg."""
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def normalize_extensions(extensions: list[str] | None) -> set[str] | None:
    if not extensions:
        return None
    return {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
    }


def build_rename_plan(
    folder: Path,
    extensions: set[str] | None,
    start: int,
    width: int,
) -> list[tuple[Path, Path]]:
    files = [path for path in folder.iterdir() if path.is_file()]
    if extensions is not None:
        files = [path for path in files if path.suffix.lower() in extensions]

    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        if path.suffix:
            grouped[path.suffix.lower()].append(path)

    plan: list[tuple[Path, Path]] = []
    for extension in sorted(grouped):
        for index, source in enumerate(sorted(grouped[extension], key=natural_key), start=start):
            number = str(index).zfill(width) if width > 0 else str(index)
            target = folder / f"{number}{extension}"
            if source.name != target.name:
                plan.append((source, target))

    return plan


def ensure_no_target_conflicts(plan: list[tuple[Path, Path]]) -> None:
    targets = [target for _, target in plan]
    duplicates = sorted({target for target in targets if targets.count(target) > 1})
    if duplicates:
        names = ", ".join(path.name for path in duplicates)
        raise ValueError(f"Duplicate target names found: {names}")

    sources = {source for source, _ in plan}
    outside_existing = [target for target in targets if target.exists() and target not in sources]
    if outside_existing:
        names = ", ".join(path.name for path in outside_existing)
        raise FileExistsError(f"Target files already exist and are not part of the rename plan: {names}")


def execute_rename(plan: list[tuple[Path, Path]]) -> None:
    token = uuid.uuid4().hex
    staged: list[tuple[Path, Path]] = []

    for index, (source, _) in enumerate(plan):
        temp = source.with_name(f".rename_tmp_{token}_{index}{source.suffix.lower()}")
        source.rename(temp)
        staged.append((temp, plan[index][1]))

    for temp, target in staged:
        temp.rename(target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename files in one folder to 1.ext, 2.ext ... for each extension."
    )
    parser.add_argument("folder", type=Path, help="Folder containing files to rename.")
    parser.add_argument(
        "--extensions",
        nargs="+",
        help="Only rename these extensions, for example: --extensions .tif .png .jpg",
    )
    parser.add_argument("--start", type=int, default=1, help="Starting number. Default: 1.")
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Zero padding width, for example --width 4 gives 0001.jpg. Default: no padding.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually rename files. Without this flag, only prints a dry-run preview.",
    )
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder does not exist: {folder}")
    if args.start < 1:
        raise ValueError("--start must be >= 1")
    if args.width < 0:
        raise ValueError("--width must be >= 0")

    extensions = normalize_extensions(args.extensions)
    plan = build_rename_plan(folder, extensions, args.start, args.width)
    ensure_no_target_conflicts(plan)

    if not plan:
        print("No files need renaming.")
        return

    print("Rename plan:")
    for source, target in plan:
        print(f"{source.name} -> {target.name}")

    if not args.execute:
        print("\nDry run only. Add --execute to rename files.")
        return

    execute_rename(plan)
    print(f"\nRenamed {len(plan)} files.")


if __name__ == "__main__":
    main()
