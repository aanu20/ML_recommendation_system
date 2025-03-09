# train_mood_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

def train_mood_model():
    # Load the preprocessed Twitter dataset
    file_path = os.path.abspath(r'Multi-domain-recommendation/data/processed_data/preprocessed_twitter_dataset.csv')
    print("Loading dataset from:", file_path)

    if os.path.exists(file_path):
        twitter_data = pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist. Please preprocess the Twitter dataset first.")

    # Features and target
    X = twitter_data[['sentiment']]  # Features (sentiment score)
    y = twitter_data['mood']  # Target (mood: Happy, Sad, Neutral)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=22)

    # Train the mood model
    mood_model = RandomForestClassifier(random_state=22)
    mood_model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = mood_model.score(X_test, y_test)
    print(f"Mood Model Accuracy: {accuracy:.2f}")

    # Save the model to a .pkl file
    output_path = os.path.abspath(r'Multi-domain-recommendation\Models\mood_model.pkl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create directory if it doesn't exist
    joblib.dump(mood_model, output_path)
    print(f"Mood model saved to {output_path}")

if __name__ == '__main__':
    train_mood_model()