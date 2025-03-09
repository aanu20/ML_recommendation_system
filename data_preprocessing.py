# Preprocess MBTI dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import re
from textblob import TextBlob
import os


# Load MBTI dataset
mbti_data = pd.read_csv(r'ML_recommendation_system\data\MBTI_dataset.csv')

# Drop unnecessary columns
mbti_data = mbti_data.drop(columns=["Gender", "Education", "Age"])

mbti_data['Personality'] = mbti_data['Personality'].str.upper()

# Normalize numeric scores
scaler = MinMaxScaler()
numeric_columns = ['Introversion Score', 'Sensing Score', 'Thinking Score', 'Judging Score']
mbti_data[numeric_columns] = scaler.fit_transform(mbti_data[numeric_columns])

mbti_data.to_csv(r'ML_recommendation_system\data\processed_data\preprocessed_mbti_dataset.csv', index=False)
print("MBTI dataset preprocessed and saved.")








def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    return text

# Function to get sentiment polarity
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Returns a score between -1 (negative) and 1 (positive)

# Function to classify mood based on sentiment score
def classify_mood(sentiment_score):
    if sentiment_score > 0.1:
        return 'Happy'
    elif sentiment_score < 0:
        return 'Sad'
    else:
        return 'Neutral'

# Load Twitter dataset
twitter_data = pd.read_csv(r'ML_recommendation_system\data\twitter_MBTI.csv')
twitter_data = twitter_data.drop(columns=["Unnamed: 0"])
twitter_data.rename(columns={"label": "Personality"}, inplace=True)
# Handle missing or invalid values in the 'text' column
twitter_data = twitter_data.dropna(subset=['text'])  # Drop rows with missing 'text'
twitter_data['text'] = twitter_data['text'].astype(str)  # Convert 'text' to strings

# Clean text data
twitter_data['text'] = twitter_data['text'].apply(clean_text)

# Perform sentiment analysis
twitter_data['sentiment'] = twitter_data['text'].apply(get_sentiment)

# Classify mood based on sentiment score
twitter_data['mood'] = twitter_data['sentiment'].apply(classify_mood)

twitter_data['Personality'] = twitter_data['Personality'].str.upper()
# Save preprocessed Twitter dataset
twitter_data.to_csv(r'ML_recommendation_system\data\processed_data\preprocessed_twitter_dataset.csv', index=False)
print("Twitter dataset preprocessed and saved.")

twitter_data = pd.read_csv(r'ML_recommendation_system\data\processed_data\preprocessed_twitter_dataset.csv')
print(twitter_data.head())

books_dataset=pd.read_csv(r"ML_recommendation_system\data\books.csv")
books_dataset=books_dataset.dropna()
books_dataset.to_csv(r"ML_recommendation_system\data\processed_data\preprocessed_books.csv")