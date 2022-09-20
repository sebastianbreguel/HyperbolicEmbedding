import os
from .functions import data_ganea
from time import sleep


def generate_data(delete_folder, create_folder) -> None:
    # Generate the folder
    if delete_folder:
        print("#" * 20, "\nDeleting folder")
        os.system("rm -rf data")
        sleep(1)
        print("Folder deleted\n")

    if create_folder:
        print("Creating folder")
        os.system("mkdir data")
        os.system("mkdir data/Prefix")
        sleep(1)
        print("\nFolder created\n")

    # run generate_data.py
    data_ganea()

    line = "\n" + "#" * 20
    print(line + "\n## Data Generated ##" + line + "\n")
