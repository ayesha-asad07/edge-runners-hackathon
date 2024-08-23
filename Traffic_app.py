import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load synthetic data
def load_synthetic_data(filename="genenrate_traffic_data (2).py"):
    with open(filename, "r") as infile:
        data = json.load(infile)
    return data

# Prepare data for training
def prepare_data(data):
    X = []
    y = []
    for sample in data:
        X.append([sample["traffic_volume"], sample["spike_detected"], sample["unauthorized_access"]])
        y.append(1 if sample["spike_detected"] or sample["unauthorized_access"] else 0)
    return np.array(X), np.array(y)

# Define and train the model
def train_phi3_model(X, y):
    model = Sequential()
    model.add(Dense(16, input_dim=X.shape[1], activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2)

    # Save the trained model as an .h5 file
    model.save("phi3_traffic_model.h5")
    print("Model trained and saved as phi3_traffic_model.h5")

    return model

if __name__ == "__main__":
    # Load data and train the model
    data = load_synthetic_data()
    X, y = prepare_data(data)
    model = train_phi3_model(X, y)
