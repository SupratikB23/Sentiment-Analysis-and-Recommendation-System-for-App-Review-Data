import re
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def cat_sentiment(x):
    if x >= 4:
        return 'positive'
    elif x <= 2:
        return 'negative'
    else:
        return 'neutral'

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_data(df, sample_size=20000, max_vocab=10000, max_len=100):
    df['sentiment'] = df['review_score'].apply(cat_sentiment)
    df['clean_text'] = df['review_text'].apply(clean_text)
    sample_df = df.sample(n=sample_size, random_state=42)

    texts = sample_df['clean_text'].values
    labels = sample_df['sentiment'].values

    le = LabelEncoder()
    y = le.fit_transform(labels)

    tokenizer = Tokenizer(num_words=max_vocab, oov_token="")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, le

def build_sentiment_model(max_vocab=10000, max_len=100, embedding_dim=100):
    model = Sequential([
        Embedding(input_dim=max_vocab, output_dim=embedding_dim, input_length=max_len),
        Bidirectional(LSTM(128, return_sequences=True)),
        Bidirectional(GRU(64, return_sequences=False)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, epochs=10, batch_size=128):
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop])
    return model, history

if __name__ == "__main__":
    apps_df = pd.read_csv("apps_info.csv", index_col=0)
    app_reviews_df = pd.read_csv("apps_reviews.csv", index_col=0)

    games_df = pd.read_csv("games_info.csv", index_col=0)
    game_reviews_df = pd.read_csv("games_reviews.csv", index_col=0)

    X_train, X_test, y_train, y_test, le_apps = prepare_data(app_reviews_df)

    app_model = build_sentiment_model()
    app_model, history = train_model(app_model, X_train, y_train)

    X_train2, X_test2, y_train2, y_test2, le_games = prepare_data(game_reviews_df)

    game_model = build_sentiment_model()
    game_model, history2 = train_model(game_model, X_train2, y_train2)
