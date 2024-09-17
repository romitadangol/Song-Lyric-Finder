from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if not already downloaded
nltk.download('stopwords')

app = Flask(__name__)

# Load stop words and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Directory containing the song lyrics
LYRICS_DIR = 'lyrics/'

# Load and preprocess song lyrics
def load_lyrics():
    documents = []
    filenames = []
    for filename in os.listdir(LYRICS_DIR):
        if filename.endswith('.txt'):
            with open(os.path.join(LYRICS_DIR, filename), 'r', encoding='utf-8') as file:
                lyrics = file.read()
                preprocessed_lyrics = preprocess_text(lyrics)
                documents.append(preprocessed_lyrics)
                filenames.append(filename)
    return documents, filenames

def preprocess_text(text):
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Remove stop words and apply stemming
    tokens = [stemmer.stem(word) for word in tokens if word.isalnum() and word.lower() not in stop_words]
    return ' '.join(tokens)

# Load and preprocess the lyrics
documents, filenames = load_lyrics()

# Vectorize the documents using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        results = search_lyrics(query)
        return render_template('index.html', query=query, results=results)
    return render_template('index.html')

def search_lyrics(query):
    # Preprocess the query
    preprocessed_query = preprocess_text(query)
    # Vectorize the query
    query_vector = vectorizer.transform([preprocessed_query])
    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    # Get the top results
    top_indices = cosine_similarities.argsort()[-5:][::-1]
    results = [(filenames[i], cosine_similarities[i]) for i in top_indices if cosine_similarities[i] > 0]
    return results

if _name_ == '__main__':
    app.run(debug=True)