import tensorflow as tf

from data import DataLoaderRPS
from model import MyModel


def main():
    TENSORBOARD_PATH = "./logs"
    CHECKPOINT_PATH = "./ckpt/cp.ckpt"

    data_loader = DataLoaderRPS()

    train_dataset = data_loader.train_dataset
    valid_dataset = data_loader.valid_dataset
    test_dataset = data_loader.test_dataset

    model = MyModel()

    # Compile the Keras model to configure the training process
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=["accuracy"],
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=TENSORBOARD_PATH, write_graph=False
    )

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    # Train model
    model.fit(
        train_dataset,
        epochs=50,
        validation_data=valid_dataset,
        callbacks=[tensorboard_callback, ckpt_callback],
    )

    # Test model
    model.load_weights(CHECKPOINT_PATH)
    test_loss, test_accuracy = model.evaluate(test_dataset)

    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")


if __name__ == "__main__":
    main()
