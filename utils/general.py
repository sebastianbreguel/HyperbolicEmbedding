import os
from .generate import generate_words
from .embedding import make_embedding
from time import sleep


def generate_data() -> None:
    # Generate the folder
    delete_older = input("Do you want to delete the older data? (y/n): ")
    if delete_older == "y":
        os.system("rm -rf data")
        sleep(1)

    new_folder = input("Do you want to create a new folder? (y/n): ")
    if new_folder == "y":
        os.system("mkdir data")
        os.system("mkdir data/distances")
        os.system("mkdir data/embeddings")
        os.system("mkdir data/words")

    # run generate_data.py
    generate_words()

    # run embedding.py
    make_embedding()
    os.system("ls data")

    print("\nDone")
