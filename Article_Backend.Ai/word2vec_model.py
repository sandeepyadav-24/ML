import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load your dataset
# Replace 'your_dataset.csv' with the path to your dataset file
df = pd.read_csv("C:/Users/hp/Documents/OpenHack/archive/data.csv")

# Choose the textual field for training the Word2Vec model
text_field = 'full_content'  # Change this to the appropriate column name from your dataset

# Preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Handle NaN values
    if pd.isnull(text):
        return ''
    
    tokens = word_tokenize(text)
    processed_text = ""
    for word in tokens:
        word = word.lower()
        if word not in stop_words:
            processed_text += lemmatizer.lemmatize(word, pos="v") + " "
    return processed_text.strip()

# Replace 'your_dataset.csv' with your actual dataset file path

# Preprocess text in the 'title' column (replace 'title' with the column containing your text data)
# Apply preprocessing to the selected text field
df['preprocessed_text'] = df[text_field].apply(preprocess_text)

# Train Word2Vec model
sentences = df['preprocessed_text'].tolist()
model = Word2Vec(sentences, vector_size=300, window=5, min_count=1, workers=4)  # Set vector_size to 300

# Save the Word2Vec model
model.save('word2vec_model.bin')
