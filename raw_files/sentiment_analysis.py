import os
from transformers import logging, pipeline
import json

# Atur lingkungan dan logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Mematikan log TensorFlow yang tidak diinginkan
logging.set_verbosity_error()  # Mematikan log Hugging Face Transformers

def analyze_sentiment(texts):
    """
    Menganalisis sentimen dari daftar kalimat.
    
    Args:
    texts (list of str): Daftar kalimat yang ingin dianalisis.
    
    Returns:
    dict: Hasil analisis sentimen dalam format JSON.
    """
    classifier = pipeline('sentiment-analysis')
    results = [classifier(text) for text in texts]
    return json.dumps(results, indent=4)

# Contoh kalimat
sentences = [
    'I am feeling great today!',
    'I am not happy with the service.',
    'This is an amazing experience!',
    'I am feeling very sad about this situation.'
]

# Menganalisis sentimen dan mencetak hasil
print(analyze_sentiment(sentences))