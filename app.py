import joblib
from textblob import TextBlob
from books_recommendation.books_recommendation import recommend_books
import pandas as pd

def main():
    # Load models
    personality_model = joblib.load(r'ML_recommendation_system/Models/personality_model.pkl')
    mood_model = joblib.load(r'ML_recommendation_system/Models/mood_model.pkl')

    # Load books dataset
    books_data = pd.read_csv(r'ML_recommendation_system/data/processed_data/preprocessed_books.csv')

    # Example user data
    user_data = {
        'Introversion Score': 0.7,
        'Sensing Score': 0.2,
        'Thinking Score': 0.8,
        'Judging Score': 0.4,
        'text': "feel good today !"
    }

    # Predict Personality
    personality = personality_model.predict([[user_data['Introversion Score'], user_data['Sensing Score'], user_data['Thinking Score'], user_data['Judging Score']]])[0]
    print(personality)
    sentiment = TextBlob(user_data['text']).sentiment.polarity
    mood = mood_model.predict([[sentiment]])[0]

    # Recommend Books
    recommended_books = recommend_books(personality, mood, books_data)
    print(recommended_books[['title', 'categories']])

if __name__ == '__main__':
    main()