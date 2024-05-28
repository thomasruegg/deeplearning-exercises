from data import DataLoaderMNIST
from model_solution import MyModel
from train import Training


def main():
    TENSORBOARD_PATH = "./logs"

    data_loader = DataLoaderMNIST(flatten=False)
    model = MyModel("MNISTClassifier")
    train = Training(model, TENSORBOARD_PATH)

    train_dataset = data_loader.train_dataset
    valid_dataset = data_loader.valid_dataset
    test_dataset = data_loader.test_dataset

    train(train_dataset, valid_dataset, test_dataset)


if __name__ == "__main__":
    main()
