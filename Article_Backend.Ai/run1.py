import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle

# Load the Word2Vec model
from gensim.models import Word2Vec

# Load the Word2Vec model
loaded_model = Word2Vec.load('C:/Users/hp/Documents/OpenHack/word2vec_model.bin')


# Define preprocessing function
def preprocess_text(text):
    if pd.isna(text):  # Check if the value is NaN
        return ''      # Return an empty string if NaN
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    tokens = word_tokenize(str(text))  # Convert to string
    processed_text = ""
    for word in tokens:
        word = word.lower()
        if word not in stop_words:
            processed_text += lemmatizer.lemmatize(word, pos="v") + " "
    return processed_text.strip()



# Load dataset
df = pd.read_csv("C:/Users/hp/Documents/OpenHack/archive/data.csv")

# Concatenate relevant text columns
df['combined_text'] = df['title'] + " " + df['description'] + " " + df['content']

# Preprocess text
df['preprocessed_text'] = df['combined_text'].apply(preprocess_text)

# Define the function for recommending articles based on input

from sklearn.feature_extraction.text import TfidfVectorizer

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



import pickle

# Save DataFrame 'df' to a pickle file
with open('df.pkl', 'wb') as f:
    pickle.dump(df, f)

# Save loaded Word2Vec model to a pickle file
with open('word2vec_model.pkl', 'wb') as f:
    pickle.dump(loaded_model, f)

# Example of usage
user_input = "climate change news in the world"
num_similar_items = 5
recommended_articles = recommend_article_based_on_input(user_input, loaded_model, df, num_similar_items)
for article in recommended_articles:
    print(f"{article['title']} :- {article['link']}")




