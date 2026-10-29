import json
import re
from pathlib import Path

import numpy as np


class TextProcessor:
    """
    Manage text data for a next-token prediction model.

    Responsibilities:
        1. Store training sentences.
        2. Tokenize human-readable text.
        3. Build and save a vocabulary.
        4. Convert tokens into numerical IDs.
        5. Prepare input and target training arrays.
    """

    def __init__(self, directory="training_data", context_size=3):
        """
        Initialize the text processor.

        Args:
            directory:
                Folder containing training data and vocabulary.

            context_size:
                Number of preceding tokens the model receives
                when predicting the next token.
        """

        if context_size < 1:
            raise ValueError("context_size must be positive.")

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        self.data_path = self.directory / "sentences.jsonl"
        self.vocab_path = self.directory / "vocabulary.json"

        self.context_size = context_size
        self.sentences = []
        self.vocabulary = {}

        self.load_data()
        self.load_vocabulary()


    def tokenize(self, text):
        """
        Convert a string into individual word tokens.

        Lowercase the text so words like 'The' and 'the'
        are treated as the same token.

        This beginner tokenizer also removes punctuation.

        Returns:
            list[str]: Individual word tokens.
        """

        return re.findall(r"\b\w+\b", text.lower())


    def add_text(self, text):
        """
        Append a new sentence to the training dataset.

        Each sentence is saved immediately to a JSONL file.

        This does not train the neural network.
        It only stores the text for future training.
        """

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Training text cannot be empty.")

        record = {"text": text.strip()}

        with self.data_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

        self.sentences.append(record["text"])


    def load_data(self):
        """
        Load previously saved training sentences.

        This restores the text corpus when the program
        starts again.
        """

        self.sentences = []

        if not self.data_path.exists():
            return

        with self.data_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    record = json.loads(line)
                    self.sentences.append(record["text"])


    def build_vocabulary(self):
        """
        Assign a permanent numerical ID to each known word.

        ID 0 is reserved for padding.
        ID 1 is reserved for unknown words.

        The vocabulary is saved and should not be rebuilt
        after a compatible model has been trained.
        """

        if self.vocabulary:
            raise ValueError(
                "A vocabulary already exists. "
                "Rebuilding it would change token IDs."
            )

        unique_words = set()

        for sentence in self.sentences:
            unique_words.update(self.tokenize(sentence))

        self.vocabulary = {
            "<PAD>": 0,
            "<UNK>": 1
        }

        for word in sorted(unique_words):
            self.vocabulary[word] = len(self.vocabulary)

        with self.vocab_path.open("w", encoding="utf-8") as file:
            json.dump(self.vocabulary, file, indent=4)


    def load_vocabulary(self):
        """
        Restore the saved token-to-ID mapping.

        This ensures that previously trained embedding
        weights still correspond to the same tokens.
        """

        if not self.vocab_path.exists():
            return

        with self.vocab_path.open("r", encoding="utf-8") as file:
            self.vocabulary = json.load(file)


    def encode(self, text):
        """
        Convert human-readable text into numerical token IDs.

        Words missing from the saved vocabulary receive
        the <UNK> token ID.
        """

        if not self.vocabulary:
            raise ValueError("Build or load a vocabulary first.")

        tokens = self.tokenize(text)

        unknown_id = self.vocabulary["<UNK>"]

        return [
            self.vocabulary.get(token, unknown_id)
            for token in tokens
        ]


    def decode(self, token_ids):
        """
        Convert numerical token IDs back into words.

        Useful for displaying model predictions
        in human-readable form.
        """

        reverse_vocabulary = {
            token_id: word
            for word, token_id in self.vocabulary.items()
        }

        return " ".join(
            reverse_vocabulary.get(int(token_id), "<UNK>")
            for token_id in token_ids
            if int(token_id) != 0
        )


    def prepare_training_data(self):
        """
        Create input and target arrays for next-token learning.

        Each input contains up to context_size preceding
        tokens, padded with zeros on the left.

        Each target contains the next token's ID.

        Returns:
            x_train:
                A 2D NumPy array of input sequences.

            y_train:
                A 1D NumPy array of correct next-token IDs.
        """

        x_train = []
        y_train = []

        for sentence in self.sentences:
            token_ids = self.encode(sentence)

            for index in range(1, len(token_ids)):
                context = token_ids[
                    max(0, index - self.context_size):index
                ]

                padding = [0] * (
                    self.context_size - len(context)
                )

                x_train.append(padding + context)
                y_train.append(token_ids[index])

        if not x_train:
            raise ValueError(
                "Not enough training text. "
                "Add sentences with at least two words."
            )

        return (
            np.array(x_train, dtype=np.int32),
            np.array(y_train, dtype=np.int32)
        )


    @property
    def vocab_size(self):
        """
        Return the total number of vocabulary tokens.

        This includes the padding and unknown tokens.
        """

        return len(self.vocabulary)