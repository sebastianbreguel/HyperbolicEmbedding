import os
from .functions import generate_words, make_embedding
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
        os.system("mkdir data/distances")
        os.system("mkdir data/embeddings")
        os.system("mkdir data/words")
        sleep(1)
        print("\nFolder created\n")

    # run generate_data.py
    generate_words()

    # run embedding.py
    make_embedding()

    line = "\n" + "#" * 20
    print(line + "\n## Data Generated ##" + line + "\n")
