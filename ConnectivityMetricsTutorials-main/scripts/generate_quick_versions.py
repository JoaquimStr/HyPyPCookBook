#!/usr/bin/env python3
"""
Generate quick versions of tutorial notebooks.

This script reads tutorial notebooks and generates "quick" versions where
inline function definitions are replaced with imports from src/.

Usage:
    python scripts/generate_quick_versions.py [notebook_path]
    python scripts/generate_quick_versions.py --all

The script looks for cells marked with:
    # [QUICK_VERSION: import_statement]

And replaces the entire cell with the specified import statement.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional


def find_quick_version_marker(source: str) -> Optional[str]:
    """
    Find the QUICK_VERSION marker in a cell source.

    Parameters
    ----------
    source : str
        The source code of a cell.

    Returns
    -------
    Optional[str]
        The import statement to use, or None if no marker found.

    Examples
    --------
    >>> find_quick_version_marker("# [QUICK_VERSION: from src.signals import generate_sine_wave]\\ndef func():")
    'from src.signals import generate_sine_wave'
    """
    pattern = r"#\s*\[QUICK_VERSION:\s*(.+?)\]"
    match = re.search(pattern, source)
    if match:
        return match.group(1).strip()
    return None


def process_cell(cell: dict) -> dict:
    """
    Process a single cell, replacing marked cells with imports.

    Parameters
    ----------
    cell : dict
        A notebook cell dictionary.

    Returns
    -------
    dict
        The processed cell (modified or unchanged).
    """
    if cell.get("cell_type") != "code":
        return cell

    source = cell.get("source", [])
    if isinstance(source, list):
        source_str = "".join(source)
    else:
        source_str = source

    import_statement = find_quick_version_marker(source_str)

    if import_statement:
        # Replace cell content with just the import
        new_cell = cell.copy()
        new_cell["source"] = [f"{import_statement}\n"]
        new_cell["outputs"] = []
        new_cell["execution_count"] = None
        return new_cell

    return cell


def generate_quick_version(notebook_path: Path) -> Path:
    """
    Generate a quick version of a tutorial notebook.

    Parameters
    ----------
    notebook_path : Path
        Path to the tutorial notebook.

    Returns
    -------
    Path
        Path to the generated quick notebook.
    """
    # Read the notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Process all cells
    processed_cells = [process_cell(cell) for cell in notebook.get("cells", [])]

    # Create the quick version notebook
    quick_notebook = notebook.copy()
    quick_notebook["cells"] = processed_cells

    # Update title in first markdown cell if present
    if processed_cells and processed_cells[0].get("cell_type") == "markdown":
        first_cell = processed_cells[0]
        source = first_cell.get("source", [])
        if isinstance(source, list):
            source_str = "".join(source)
        else:
            source_str = source

        # Add "(Quick Version)" to the title if not already there
        if "# " in source_str and "(Quick Version)" not in source_str:
            updated_source = re.sub(
                r"^(# .+?)(\n|$)",
                r"\1 (Quick Version)\2",
                source_str,
                count=1
            )
            first_cell["source"] = [updated_source]

    # Generate output path
    stem = notebook_path.stem
    quick_path = notebook_path.parent / f"{stem}_quick.ipynb"

    # Write the quick version
    with open(quick_path, "w", encoding="utf-8") as f:
        json.dump(quick_notebook, f, indent=1, ensure_ascii=False)

    return quick_path


def find_all_notebooks(base_path: Path) -> list[Path]:
    """
    Find all tutorial notebooks (excluding quick versions).

    Parameters
    ----------
    base_path : Path
        Base path to search from.

    Returns
    -------
    list[Path]
        List of notebook paths.
    """
    notebooks = []
    for notebook_path in base_path.rglob("*.ipynb"):
        # Skip quick versions, checkpoints, and hidden files
        if "_quick" in notebook_path.stem:
            continue
        if ".ipynb_checkpoints" in str(notebook_path):
            continue
        if notebook_path.name.startswith("."):
            continue
        notebooks.append(notebook_path)
    return sorted(notebooks)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate quick versions of tutorial notebooks."
    )
    parser.add_argument(
        "notebook",
        nargs="?",
        help="Path to a specific notebook to process"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all notebooks in the notebooks/ directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing files"
    )

    args = parser.parse_args()

    # Determine project root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    if args.all:
        notebooks_dir = project_root / "notebooks"
        if not notebooks_dir.exists():
            print(f"Error: notebooks directory not found at {notebooks_dir}")
            return 1
        notebooks = find_all_notebooks(notebooks_dir)
    elif args.notebook:
        notebook_path = Path(args.notebook).resolve()
        if not notebook_path.exists():
            print(f"Error: notebook not found at {notebook_path}")
            return 1
        notebooks = [notebook_path]
    else:
        parser.print_help()
        return 1

    print(f"Found {len(notebooks)} notebook(s) to process\n")

    for notebook_path in notebooks:
        try:
            relative_path = notebook_path.relative_to(project_root)
        except ValueError:
            relative_path = notebook_path.name
        print(f"Processing: {relative_path}")

        if args.dry_run:
            quick_path = notebook_path.parent / f"{notebook_path.stem}_quick.ipynb"
            try:
                display_path = quick_path.relative_to(project_root)
            except ValueError:
                display_path = quick_path.name
            print(f"  Would create: {display_path}")
        else:
            try:
                quick_path = generate_quick_version(notebook_path)
                try:
                    display_path = quick_path.relative_to(project_root)
                except ValueError:
                    display_path = quick_path.name
                print(f"  Created: {display_path}")
            except Exception as e:
                print(f"  Error: {e}")
                continue

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
