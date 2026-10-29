from processing.TextProcessor import TextProcessor
from models.text_model import TextModel


def main():
    """
    Coordinate text processing and neural network training.

    Loads saved training data and vocabulary, creates or
    loads the neural network, then trains it.
    """

    processor = TextProcessor(
        directory="training_data",
        context_size=3
    )

    # Add example sentences on the first run only.
    if not processor.sentences:
        processor.add_text("The sky is blue")
        processor.add_text("The ocean is blue")
        processor.add_text("The grass is green")
        processor.add_text("The sky is beautiful")

    # Create the vocabulary if one has not been saved.
    if not processor.vocabulary:
        processor.build_vocabulary()

    # Convert all saved sentences into training arrays.
    x_train, y_train = processor.prepare_training_data()

    # Initialize the neural network.
    ai = TextModel(vocab_size=processor.vocab_size)

    # Reuse the existing model when available.
    if not ai.model_load("text_model"):
        ai.create_model()

    # Train using the prepared data.
    ai.model_train(x_train, y_train, epochs=100)

    # Save a new checkpoint without overwriting the original.
    ai.model_save("text_model_updated")

    user_input = input("Type a word you'd like my model to guess")
    
    encoded_input = processor.encode(user_input)
    
    predicted_id = ai.prediction(encoded_input)
    
    predicted_word = processor.decode([predicted_id])
    
    print(f"Predicted next word: {predicted_word}")



if __name__ == "__main__":
    main()