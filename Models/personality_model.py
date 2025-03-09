# train_personality_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the preprocessed MBTI dataset
mbti_data = pd.read_csv(r'ML_recommendation_system\data\processed_data\preprocessed_mbti_dataset.csv')

# Features and target
X = mbti_data[['Introversion Score', 'Sensing Score', 'Thinking Score', 'Judging Score']]
y = mbti_data['Personality']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the personality model
personality_model = RandomForestClassifier(random_state=42)
personality_model.fit(X_train, y_train)

# Evaluate the model
accuracy = personality_model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model to a .pkl file
joblib.dump(personality_model, r'ML_recommendation_system\Models\personality_model.pkl')
print("Personality model saved to personality_model.pkl")