import tensorflow as tf

from data import DataLoaderIMDB
from model import MyModel


def main():
    TENSORBOARD_PATH = "./logs"

    data_loader = DataLoaderIMDB()

    train_dataset = data_loader.train_dataset
    valid_dataset = data_loader.valid_dataset
    test_dataset = data_loader.test_dataset
    unsupervised_dataset = data_loader.unsupervised_dataset

    # Create vocabulary for text vectorizer
    vocab_1 = train_dataset.map(lambda text, label: text)
    vocab_2 = unsupervised_dataset.map(lambda text, label: text)
    vocabulary = vocab_1.concatenate(vocab_2)

    model = MyModel(vocabulary)

    # Compile the Keras model to configure the training process
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["accuracy"],
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=TENSORBOARD_PATH, write_graph=False
    )

    # Train model
    model.fit(
        train_dataset,
        epochs=10,
        validation_data=valid_dataset,
        callbacks=[tensorboard_callback],
    )

    # Test model
    test_loss, test_accuracy = model.evaluate(test_dataset)

    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")


if __name__ == "__main__":
    main()
