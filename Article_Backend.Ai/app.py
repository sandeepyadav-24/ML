from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

# Load df from pickle file
with open('df.pkl', 'rb') as f:
    df = pickle.load(f)

# Load loaded_model from pickle file
with open('word2vec_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

def recommend_article_based_on_input(user_input, loaded_model, df, num_similar_items):
    # Concatenate relevant text columns
    combined_text = df['title'] + " " + df['description'] + " " + df['content']
    
    # Calculate TF-IDF vectors for combined text
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_text.astype(str))
    
    # Calculate TF-IDF vector for user input
    user_tfidf_vector = tfidf_vectorizer.transform([user_input])
    
    # Calculate cosine similarity between user input and combined text
    cosine_similarities = pairwise_distances(tfidf_matrix, user_tfidf_vector, metric='cosine').ravel()
    
    # Get indices of most similar articles
    indices = np.argsort(cosine_similarities)[:num_similar_items]

    # Extract recommended articles information
    recommended_articles = []
    for index in indices:
        article_info = {
            'title': df['title'][index],
            'link': df['url'][index],
            'description': df['description'][index],
            'author': df['author'][index],
            'year': pd.to_datetime(df['published_at'][index]).year  # Extract year from published_at
        }
        recommended_articles.append(article_info)

    return recommended_articles

# Flask App
app = Flask(__name__)

@app.route('/recommend_articles', methods=['POST'])
def recommend_articles():
    data = request.get_json()
    user_input = data['user_input']
    # num_similar_items = data['num_similar_items']
    recommended_articles = recommend_article_based_on_input(user_input, loaded_model, df,5)
    return jsonify(recommended_articles)

if __name__ == '__main__':
    app.run(debug=True)


