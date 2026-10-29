
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="TextAI")
class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    """
    Convert token IDs into embeddings and add position information.
    """

    def __init__(self, vocab_size, context_size, embed_dim, **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embed_dim = embed_dim

        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim
        )

        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=context_size,
            output_dim=embed_dim
        )


    def call(self, token_ids):
        positions = tf.range(
            start=0,
            limit=tf.shape(token_ids)[1]
        )

        return (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
        )


    def get_config(self):
        config = super().get_config()

        config.update({
            "vocab_size": self.vocab_size,
            "context_size": self.context_size,
            "embed_dim": self.embed_dim
        })

        return config


@tf.keras.utils.register_keras_serializable(package="TextAI")
class TransformerBlock(tf.keras.layers.Layer):
    """
    Transformer decoder block with causal self-attention.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads
        )

        self.feed_forward = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim)
        ])

        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()


    def call(self, inputs, token_ids, training=None):
        # Mask padding tokens so attention ignores them.
        padding_mask = tf.not_equal(token_ids, 0)
        attention_mask = padding_mask[:, tf.newaxis, :]

        # Causal masking prevents looking ahead.
        attention_output = self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=attention_mask,
            use_causal_mask=True,
            training=training
        )

        x = self.norm1(inputs + attention_output)

        feed_forward_output = self.feed_forward(
            x,
            training=training
        )

        return self.norm2(x + feed_forward_output)


    def get_config(self):
        config = super().get_config()

        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim
        })

        return config


@tf.keras.utils.register_keras_serializable(package="TextAI")
class LastToken(tf.keras.layers.Layer):
    """
    Select the last position's representation.

    The TextProcessor left-pads sequences, so the final
    position always contains a real token.
    """

    def call(self, inputs):
        return inputs[:, -1, :]