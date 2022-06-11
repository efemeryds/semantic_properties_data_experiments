

def create_subdataset(amount):
    count = 0
    with open("data.txt") as infile:
        for line in infile:
            with open('piece_of_data.txt', 'a') as f:
                f.write(line)
            count += 1
            if count > amount:
                break


if __name__ == "__main__":
    create_subdataset(30000)

