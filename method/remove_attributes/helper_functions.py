import os


def create_all_concepts_file():
    dir_path = "concepts_lists/"
    all_files = os.listdir(dir_path)

    for file in all_files:
        with open(f"{dir_path}{file}") as infile:
            for line in infile:
                with open(f'{dir_path}/all_concepts.txt', 'a') as f:
                    f.write(line)


if __name__ == "__main__":
    # create_all_concepts_file()
    pass