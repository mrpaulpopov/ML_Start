import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def load_and_prepare_data(path):
    movies = pd.read_csv(path)
    # Relabeling the 'sentiment' column as 0's and 1's
    movies['sentiment'] = movies['sentiment'].map({'positive': 1, 'negative': 0})
    return movies


def vectorize_text(text, item=None, max_features=2000):
    # Create word features (limit to top 5000 words for efficiency)
    # CountVectorizer creates Bag of words.
    # 2000 the most frequent words (recommended value)
    # 'stop_words = english' deletes english propositions and articles.
    vectorizer = CountVectorizer(max_features=max_features, stop_words='english', binary=True)
    if item is not None:
        X = vectorizer.fit_transform(text[item])
    else:
        X = vectorizer.fit_transform(text)
    return X, vectorizer


def train_model(X, y):
    # Train logistic regression (linear model for interpretable weights)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def get_top_words(model, vectorizer):
    # Get feature names (words) and their coefficients
    feature_names = vectorizer.get_feature_names_out()
    word_weights = model.coef_[0]  # Weights for positive sentiment

    # Create DataFrame of words and their sentiment scores
    word_sentiments = pd.DataFrame({
        'word': feature_names,
        'weight': word_weights
    })

    # Sort words by sentiment strength
    most_positive = word_sentiments.sort_values('weight', ascending=False).head(10)
    most_negative = word_sentiments.sort_values('weight').head(10)
    return most_positive, most_negative


def plot_top_words(most_positive, most_negative):
    # Plot top positive/negative words
    plt.figure(figsize=(10, 6))

    # Positive words
    plt.subplot(121)
    plt.barh(most_positive['word'], most_positive['weight'], color='green')
    plt.gca().invert_yaxis() # invert graph
    plt.title('Top Positive Words')
    plt.xlabel('Sentiment Weight')

    # Negative words
    plt.subplot(122)
    plt.barh(most_negative['word'], most_negative['weight'], color='red')
    plt.gca().invert_yaxis() # invert graph
    plt.title('Top Negative Words')
    plt.xlabel('Sentiment Weight')

    plt.tight_layout()
    plt.savefig('images/01.png')
    plt.show()


def find_extreme_reviews(model, X, movies):
    # Predict sentiment scores for all reviews
    predictions = model.predict_proba(X)[:, 1]  # Probability of positive sentiment

    # Add scores to the DataFrame
    movies['predictions'] = predictions

    # Find the most positive and negative reviews
    most_positive_review = movies.sort_values('predictions', ascending=False).head(1)
    most_negative_review = movies.sort_values('predictions').head(1)
    return most_positive_review, most_negative_review


def predict_own_review(model, vectorizer):
    print("Enter your review")
    input_review = [input()]
    new_X = vectorizer.transform(input_review)
    y = model.predict(new_X)
    y_proba = model.predict_proba(new_X)[:, 1]
    if y == 1:
        print(f"It's a positive review with a probability of {y_proba[0]*100:.2f}%.")
    else:
        y_proba = 1 - y_proba
        print(f"It's a negative review with a probability of {y_proba[0]*100:.2f}%.")


def main():
    movies = load_and_prepare_data('data/IMDB_Dataset.csv')
    print('Data Head:')
    print(movies.head())
    print('\nData Count:')
    print(movies.count())

    y = movies['sentiment']
    X, vectorizer = vectorize_text(movies, item='review', max_features=2000)
    model = train_model(X, y)
    most_positive, most_negative = get_top_words(model, vectorizer)

    print('\nMost Positive Words:')
    print(most_positive)
    print('\nMost Negative Words:')
    print(most_negative)

    plot_top_words(most_positive, most_negative)
    most_positive_review, most_negative_review = find_extreme_reviews(model, X, movies)

    print("\nMost Positive Review:")
    print(most_positive_review['review'].iloc[0][:250]+'...')

    print("\nMost Negative Review:")
    print(most_negative_review['review'].iloc[0][:250]+'...')

    predict_own_review(model, vectorizer)


if __name__ == "__main__":
    main()