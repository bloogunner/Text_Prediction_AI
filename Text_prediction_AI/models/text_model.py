import tensorflow as tf
from pathlib import Path
import numpy as np


class TextModel:

    def __init__(self, vocab_size):
        """
        Initialize the text model class.

        vocab_size:
            Number of unique tokens in the vocabulary.

        self.model:
            Stores the TensorFlow neural network.
            Initially None until created or loaded.
        """

        self.vocab_size = vocab_size
        self.model = None


    def create_model(self):
        """
        Create a neural network for next-token prediction.

        Embedding:
            Converts token IDs into learned vectors.

        GRU:
            Processes sequences and maintains a hidden state.

        Dense:
            Produces probabilities for the next token.
        """

        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(None,)),

            tf.keras.layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=32,
                mask_zero=True
            ),

            tf.keras.layers.GRU(64),

            tf.keras.layers.Dense(
                self.vocab_size,
                activation="softmax"
            )
        ])

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        return self.model


    def model_train(self, x_train, y_train, epochs=100):
        """
        Train the neural network using tokenized text.

        x_train:
            Input sequences containing token IDs.

        y_train:
            Correct next-token IDs.

        epochs:
            Number of times the model processes
            the complete training dataset.
        """

        if self.model is None:
            raise ValueError("Create or load a model first.")

        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            verbose=1
        )

        return history


    def model_save(self, model_name, overwrite=False):
        """
        Save the model to the current working directory.

        If a model already exists, do not overwrite it.
        """

        if self.model is None:
            raise ValueError("No model available to save.")


        model_path = Path.cwd() / f"{model_name}.keras"

        if model_path.exists() and not overwrite:
            print(f"Model already exists: {model_path}")

            return None

        self.model.save(str(model_path), overwrite=overwrite)

        print(f"Model saved to: {model_path}")


    def model_load(self, model_name):
        """
        Load a previously saved TensorFlow model.

        The loaded model replaces self.model so it can
        be used for training or predictions.
        """

        current_directory = Path.cwd()

        model_path = current_directory / f"{model_name}.keras"

        if not model_path.is_file():
            print("No saved model found.")

            return False

        self.model = tf.keras.models.load_model(
            str(model_path)
        )

        print(f"Model loaded from: {model_path}")

        return True
    
    
    def prediction(self, prediction):
        '''
        Generate a prediction using the trained model.
        
        Args:
            prediction:
                    A sequence of numerical token IDs
                    supplied by the user
                    
                    
            
        Returns:
            int:
                The predicted next-token ID.
        '''
        
        if self.model is None:
            raise ValueError("No model has been created or loaded.")
        
        '''Convert the user's input into a NumPy array.'''
        input_data = np.array(
            [prediction],
            dtype=np.int32
        )
        
        '''Ask the neural netowrk to calculate probabilites.'''
        probablities = self.model.predict(
            input_data,
            verbose=1
        )
        
        '''Find the token with the highest probability.'''
        predicted_token = np.argmax(probablities[0])
        
        return int(predicted_token)
        