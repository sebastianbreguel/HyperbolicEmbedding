import os

# * File to run the three main models


def main():
    print("Hello World")
    # os.system("python3 main.py --gen_data ")

    os.system("python3 main.py --make_train_eval --model euclidean --optimizer Adam")

    os.system("python3 main.py --make_train_eval --model hyperbolic --optimizer Adam")


if __name__ == "__main__":
    main()
