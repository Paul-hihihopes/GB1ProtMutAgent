"""运行完整实验流程，并保存包含输出的 Notebook。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["api", "offline", "auto"], default="auto")
    parser.add_argument("--kernel", default="python3", help="用于执行实验的 Jupyter 内核名称")
    parser.add_argument("--output-dir", type=Path, help="结果、图片、模型及已执行 Notebook 的输出目录")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ["LLM_BACKEND"] = args.backend
    if args.output_dir:
        os.environ["EXPERIMENT_OUTPUT"] = str(args.output_dir.resolve())
    output_root = Path(os.environ.get("EXPERIMENT_OUTPUT", str(root))).resolve()
    notebook_path = root / "notebooks" / "protein_directed_evolution.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    destination = output_root / "notebooks" / notebook_path.name
    destination.parent.mkdir(parents=True, exist_ok=True)

    def progress(cell, cell_index, **kwargs):
        title = cell.source.splitlines()[0] if cell.source else ""
        print(f"[{cell_index + 1}/{len(notebook.cells)}] {title[:90]}", flush=True)

    client = NotebookClient(notebook, timeout=1800, kernel_name=args.kernel,
                            resources={"metadata": {"path": str(root)}},
                            on_cell_start=progress)
    try:
        client.execute()
    except Exception:
        nbformat.write(notebook, destination.with_name("experiment_incomplete.ipynb"))
        raise
    nbformat.write(notebook, destination)
    print(f"实验完成：{destination}", flush=True)


if __name__ == "__main__":
    main()
