# Sentiment Analysis with Recurrent Neural Networks

This project leverages Recurrent Neural Networks (RNNs) with Keras to perform sentiment analysis on text data. 

By analyzing the sentiment behind texts, this tool can differentiate between positive and negative sentiments, making it invaluable for understanding user feedback, social media monitoring, and more.

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)

## Description

Sentiment Analysis with RNNs utilizes the power of LSTM (Long Short-Term Memory) networks to process and interpret sequences of text. 

The project is designed to be accessible and easy to use, requiring only a basic understanding of Python and machine learning concepts. 

## Installation

To get started with this project, clone the repo and install the required dependencies:

```bash
git clone https://github.com/Sorena-Dev/Sentiment-Analysis-with-Recurrent-Networks.git
cd Sentiment-Analysis-with-Recurrent-Networks
pip install numpy keras tensorflow sklearn
```

## Usage

To use the sentiment analysis model, follow these steps:

1. Prepare your dataset with texts and their corresponding sentiments.
2. Use the provided `DataPreparation` class to tokenize and pad your text data.
3. Initialize and train the `SentimentAnalysisModel` with your dataset.
4. Use the `PredictSentiment` class to predict the sentiment of new sentences.

Example:

```python
from Sentiment_Analysis_with_Recurrent_Neural_Networks import main
main()
```

This will train the model with the predefined texts and print the sentiment of the new sentence.

## Features

- **Data Preparation**: Tokenizes and pads text data for model input.
- **Model Definition**: Defines an LSTM-based RNN model for sentiment analysis.
- **Training and Validation**: Supports splitting data into training and validation sets.
- **Sentiment Prediction**: Predicts the sentiment of new text inputs.
