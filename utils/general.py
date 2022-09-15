import os

if __name__ == "__main__":

    # Generate the folder
    os.system("mkdir -p ../data")

    # run generate_data.py
    os.system("python3 generate_data.py")

    # run embedding.py
    os.system("python3 embedding.py")

    print("Done")
