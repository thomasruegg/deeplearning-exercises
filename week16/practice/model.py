import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    TextVectorization,
    Embedding,
    Bidirectional,
    GRU,
    Dense,
)


class MyModel(Model):
    """
    RNN for text classification of movie reviews.

    Parameters
    ----------
    vocabulary : ndarray (None,)
        Training vocabulary for text vectorizer
    """

    def __init__(self, vocabulary, **kwargs):
        super(MyModel, self).__init__(**kwargs)

        self.enc_1 = TextVectorization(max_tokens=1000)
        self.enc_1.adapt(vocabulary)
        self.emb_1 = Embedding(
            input_dim=len(self.enc_1.get_vocabulary()), output_dim=64, mask_zero=True
        )
        # TODO: Instantiate additional layers

    def call(self, x, training=False):
        """
        Forward pass through RNN.

        Parameters
        ----------
        x : ndarray (None,)
            Mini-batch of byte strings which corresponds to RNN input
        training : bool, optional
            Training or testing mode, by default False

        Returns
        -------
        Tensor float32 (None, 1)
            Logits of classifier (pre-sigmoid activations)
        """
        t_enc_1 = self.enc_1(x)
        t_emb_1 = self.emb_1(t_enc_1)

        # TODO: Add Bidirectional GRU and Dense layers
        out = t_emb_1

        return out


if __name__ == "__main__":
    model = MyModel(["bla", "blu"])
