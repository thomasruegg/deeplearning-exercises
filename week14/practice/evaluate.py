import cv2
import numpy as np
import tensorflow as tf

from model import MyModel


# Camera context handler
class MyVideoCapture(cv2.VideoCapture):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


def main():
    CHECKPOINT_PATH = "./ckpt/cp.ckpt"
    DEVICE_INDEX = 0
    IMAGE_SHAPE = (300, 300, 3)
    LEGEND = ["Rock", "Paper", "Scissors"]

    # Load neural network
    classifier = MyModel()
    classifier.load_weights(CHECKPOINT_PATH)

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    # Fetch images in loop
    with MyVideoCapture(DEVICE_INDEX) as cap:
        while True:
            # Fetch frame
            _, frame = cap.read()

            # Preprocess image
            limit_top = (frame.shape[0] // 2) - (IMAGE_SHAPE[0] // 2)
            limit_bottom = (frame.shape[0] // 2) + (IMAGE_SHAPE[0] // 2)
            limit_left = (frame.shape[1] // 2) - (IMAGE_SHAPE[1] // 2)
            limit_right = (frame.shape[1] // 2) + (IMAGE_SHAPE[1] // 2)
            image = frame[limit_top:limit_bottom, limit_left:limit_right, :]

            # Classify image
            out = classifier.predict(image[np.newaxis, ...])
            class_idx = tf.math.argmax(out, axis=1).numpy()[0]
            classified_image = cv2.putText(
                image,
                str(LEGEND[class_idx]),
                (40, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                3,
                cv2.LINE_AA,
            )

            # Display classified_image
            cv2.imshow("image", classified_image)

            WAIT_MS = 5
            ESC_IN_ASCII = 27
            if cv2.waitKey(WAIT_MS) == ESC_IN_ASCII:
                break


if __name__ == "__main__":
    main()
