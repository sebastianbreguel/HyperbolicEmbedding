import argparse
from data_gen import generate_data

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
