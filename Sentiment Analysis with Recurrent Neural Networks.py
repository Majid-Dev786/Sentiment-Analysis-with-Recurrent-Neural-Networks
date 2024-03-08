import numpy as np
from typing import List, Tuple
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

TEXTS = ['I love this product!', 'I hate this product!', 'Awesome item!', 'Not happy with this!']
SENTIMENTS = [1, 0, 1, 0]
EMBEDDING_DIM = 10
NEW_SENTENCE = "I love this product!"

class DataPreparation:
    """Prepare and preprocess text data for sentiment analysis."""
    MAX_SEQUENCE_LENGTH = 0
    VOCAB_SIZE = 0

    def __init__(self, texts: List[str], sentiments: List[int]) -> None:
        self.texts = texts
        self.sentiments = sentiments
        self.tokenizer = Tokenizer()

    def tokenize_and_pad_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Tokenize and pad text data. Update MAX_SEQUENCE_LENGTH and VOCAB_SIZE based on the data."""
        self.tokenizer.fit_on_texts(self.texts)
        sequences = self.tokenizer.texts_to_sequences(self.texts)
        DataPreparation.MAX_SEQUENCE_LENGTH = max([len(sequence) for sequence in sequences])
        data = pad_sequences(sequences, maxlen=DataPreparation.MAX_SEQUENCE_LENGTH)
        DataPreparation.VOCAB_SIZE = len(self.tokenizer.word_index) + 1
        return data, np.array(self.sentiments)

class SentimentAnalysisModel:
    """Define and train a sentiment analysis model."""

    def __init__(self, vocab_size: int, embedding_dim: int, sequence_length: int) -> None:
        self.model = Sequential()
        self.model.add(Embedding(vocab_size, embedding_dim, input_length=sequence_length))
        self.model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(1, activation='sigmoid'))

    def compile_model(self) -> None:
        """Compile the model with a specific loss function, optimizer, and metrics."""
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self, train_data: np.ndarray, train_labels: np.ndarray, validation_data: np.ndarray, validation_labels: np.ndarray) -> None:
        """Train the sentiment analysis model."""
        self.model.fit(train_data, train_labels, epochs=10, validation_data=(validation_data, validation_labels), verbose=2)

class PredictSentiment:
    """Predict sentiment based on a trained model and tokenizer."""
    def __init__(self, model: Sequential, tokenizer: Tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, sentence: str) -> None:
        """Predict sentiment of a sentence and print the result."""
        sequence = self.tokenizer.texts_to_sequences([sentence])
        padded_sequence = pad_sequences(sequence, maxlen=DataPreparation.MAX_SEQUENCE_LENGTH)
        prediction = self.model.predict(padded_sequence)
        sentiment = "positive" if prediction > 0.5 else "negative"
        print(f"The sentiment of the sentence is {sentiment}")

def main() -> None:
    """Main function to prepare data, train the model, and predict sentiment."""
    data_prep = DataPreparation(TEXTS, SENTIMENTS)
    data, labels = data_prep.tokenize_and_pad_data()

    train_data, validation_data, train_labels, validation_labels = train_test_split(data, labels, test_size=0.2)

    sentiment_model = SentimentAnalysisModel(DataPreparation.VOCAB_SIZE, EMBEDDING_DIM, DataPreparation.MAX_SEQUENCE_LENGTH)
    sentiment_model.compile_model()
    sentiment_model.train_model(train_data, train_labels, validation_data, validation_labels)

    sentiment_predictor = PredictSentiment(sentiment_model.model, data_prep.tokenizer)
    sentiment_predictor.predict(NEW_SENTENCE)

if __name__ == "__main__":
    main()