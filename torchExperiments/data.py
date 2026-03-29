"""Entry point for dataset generation.

Usage
-----
    # Generate ganea prefix data (creates data/Prefix/ folder):
    python data.py --task ganea --replace 0.5 --create_folder

    # Generate mircea phylogenetic data:
    python data.py --task mircea

    # Download MNIST and save UMAP-reduced embeddings:
    python data.py --task MNIST
"""

from __future__ import annotations

import argparse

from training import generate_data

if "__main__" == __name__:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--replace", type=float, help="replace for a prefix", default=0.5
    )

    parser.add_argument(
        "--create_folder", action="store_true", help="Create data folder"
    )
    parser.add_argument("--task", action="store", help="task to gen the data")

    args = parser.parse_args()
    create_folder = args.create_folder
    replace = args.replace
    task = args.task

    generate_data(create_folder, replace, task)

    print("#" * 22 + "\n### Data generated ###\n" + "#" * 22)
