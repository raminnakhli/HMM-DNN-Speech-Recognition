
def train_set_loader():
    f = open('data/Train_Arabic_Digit.txt', 'r')
    plain_data = [line.split() for line in f]

    train_set = []
    length = []
    sample = []

    for i in range(len(plain_data) - 1):
        if not plain_data[i + 1]:
            train_set.append(sample)
            length.append(len(sample))
            sample = []
        else:
            sample.append([float(j) for j in plain_data[i + 1]])

    train_set.append(sample)
    length.append(len(sample))

    return train_set, length


def test_set_loader():
    f = open('data/Test_Arabic_Digit.txt', 'r')
    plain_data = [line.split() for line in f]

    test_set = []
    length = []
    sample = []

    for i in range(len(plain_data) - 1):
        if not plain_data[i + 1]:
            test_set.append(sample)
            length.append(len(sample))
            sample = []
        else:
            sample.append([float(j) for j in plain_data[i + 1]])

    test_set.append(sample)
    length.append(len(sample))

    return test_set, length


def dataset_loader():
    train_set, train_len = train_set_loader()
    test_set, testn_len = test_set_loader()
    return train_set, train_len, test_set, testn_len
