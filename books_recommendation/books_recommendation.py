import pandas as pd
def recommend_books(personality, mood, books_data):
    """
    Recommend books based on personality and mood.

    Parameters:
        personality (str): The user's personality type (e.g., 'INTJ', 'ENFP').
        mood (str): The user's current mood (e.g., 'Happy', 'Sad', 'Neutral').
        books_data (pd.DataFrame): The dataset containing book information.

    Returns:
        pd.DataFrame: A subset of books_data that matches the recommendation criteria.
    """
    # INTJ: Introverted, Intuitive, Thinking, Judging
    if personality == 'INTJ':
        if mood == 'Happy':
            return books_data[books_data['categories'].str.contains('Music|Fiction', case=False, na=False)]
        elif mood == 'Sad':
            return books_data[books_data['categories'].str.contains('inspirational|self-help', case=False, na=False)]
        elif mood == 'Neutral':
            return books_data[books_data['categories'].str.contains('science|technology', case=False, na=False)]

    # ENFP: Extraverted, Intuitive, Feeling, Perceiving
    elif personality == 'ENFP' or "INFP":
        if mood == 'Happy':
            return books_data[books_data['categories'].str.contains('adventure|creative', case=False, na=False)]
        elif mood == 'Sad':
            return books_data[books_data['categories'].str.contains('comforting|emotional', case=False, na=False)]
        elif mood == 'Neutral':
            return books_data[books_data['categories'].str.contains('philosophy|spirituality', case=False, na=False)]
        
    elif personality == 'ESFP':
        if mood == 'Happy':
            return books_data[books_data['categories'].str.contains('inspirational|Art',case=False, na=False)]
        elif mood == 'Sad':
            return books_data[books_data['categories'].str.contains('Travel|spirituality|culinary',case=False, na=False)]
        elif mood == 'Neutral':
            return books_data[books_data['categories'].str.contains('philosophy|Music', case=False, na=False)]

    elif personality == 'ISTJ':
        if mood == 'Happy':
            return books_data[books_data['categories'].str.contains('historical|non-fiction',case=False, na=False)]
        elif mood == 'Sad':
            return books_data[books_data['categories'].str.contains('motivational|practical',case=False, na=False)]
        elif mood == 'Neutral':
            return books_data[books_data['categories'].str.contains('business|finance',case=False, na=False)]

    # ISFP: Introverted, Sensing, Feeling, Perceiving
    elif personality == 'ISFP':
        if mood == 'Happy':
            return books_data[books_data['categories'].str.contains('Art|nature',case=False, na=False)]
        elif mood == 'Sad':
            return books_data[books_data['categories'].str.contains('poetry|Travel',case=False, na=False)]
        elif mood == 'Neutral':
            return books_data[books_data['categories'].str.contains('spirituality|culinary',case=False, na=False)]
        
    elif personality == 'ENTJ':
        if mood == 'Happy':
            return books_data[books_data['categories'].str.contains('Fiction|nature',case=False, na=False)]
        elif mood == 'Sad':
            return books_data[books_data['categories'].str.contains('Travel|Music', case=False, na=False)]
        elif mood == 'Neutral':
            return books_data[books_data['categories'].str.contains('business|finance',case=False, na=False)]
    elif personality == 'ISTP':
        if mood == 'Happy':
            return books_data[books_data['categories'].str.contains('Fiction|historical',case=False, na=False)]
        elif mood == 'Sad':
            return books_data[books_data['categories'].str.contains('self-help|Music', case=False, na=False)]
        elif mood == 'Neutral':
            return books_data[books_data['categories'].str.contains('business|inspirational',case=False, na=False)]
    else:
        # Default recommendation for unknown personality or mood
        return books_data[books_data['categories'].str.contains('bestseller|popular', case=False, na=False)]

# Example usage
if __name__ == '__main__':
    # Load the books dataset
    books_data = pd.read_csv(r'ML_recommendation_system\data\processed_data\preprocessed_books.csv')

    # Example input
    personality = 'INTJ'
    mood = 'Neutral'

    # Get recommended books
    recommended_books = recommend_books(personality, mood, books_data)
    if recommended_books.empty:
        print("No books match the criteria. Here are some popular books:")
        print(books_data[books_data['categories'].str.contains('bestseller|popular', case=False, na=False)][['title', 'categories']])
    else:
        print(recommended_books[['title', 'categories']])



