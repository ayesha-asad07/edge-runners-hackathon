import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the generated data
def load_data():
    with open('synthetic_login_data.json', 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def preprocess_data(df):
    # Encode categorical variables like location and device
    label_encoders = {}
    
    for column in ['location', 'device']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    X = df[['timestamp', 'location', 'device']]  # Features
    y = df['label']  # Target
    
    return X, y, label_encoders

def train_model(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return clf

if __name__ == "__main__":
    data = load_data()
    X, y, label_encoders = preprocess_data(data)
    model = train_model(X, y)
