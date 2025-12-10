from flask import Flask, render_template, request, jsonify
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import numpy as np
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# ============================================
# LOAD MODELS
# ============================================

print("Loading models...")

# Extractive Summarizer
class ExtractiveSummarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def calculate_similarity(self, sent1, sent2):
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([sent1, sent2])
            similarity = (tfidf_matrix * tfidf_matrix.T).toarray()[0][1]
            return similarity
        except:
            return 0
    
    def summarize(self, text, num_sentences=3):
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    similarity_matrix[i][j] = self.calculate_similarity(
                        sentences[i], sentences[j]
                    )
        
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        
        ranked_sentences = sorted(
            ((scores[i], s) for i, s in enumerate(sentences)), 
            reverse=True
        )
        
        summary_sentences = sorted(
            ranked_sentences[:num_sentences], 
            key=lambda x: sentences.index(x[1])
        )
        
        summary = ' '.join([s[1] for s in summary_sentences])
        return summary

# Initialize models
extractive_model = ExtractiveSummarizer()
abstractive_model = pipeline("summarization", model="facebook/bart-large-cnn")

print("Models loaded successfully!")

# ============================================
# ROUTES
# ============================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.json
        text = data.get('text', '')
        method = data.get('method', 'both')
        num_sentences = data.get('num_sentences', 3)
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        results = {}
        
        # Extractive summarization
        if method in ['extractive', 'both']:
            ext_summary = extractive_model.summarize(text, num_sentences=int(num_sentences))
            results['extractive'] = ext_summary
        
        # Abstractive summarization
        if method in ['abstractive', 'both']:
            abs_summary = abstractive_model(
                text, 
                max_length=130, 
                min_length=30, 
                do_sample=False
            )
            results['abstractive'] = abs_summary[0]['summary_text']
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)