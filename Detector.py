import numpy as np
import pandas as pd
import joblib
import sys
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.base import BaseEstimator, TransformerMixin
from url_utils import extract_url_features
# Download necessary NLTK data if not present
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK data (stopwords, punkt, wordnet)...")
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("Downloads complete.")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """This function must be identical to the one used during training."""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(lemmas)

# --- All Helper functions for feature engineering ---
def get_sender_domain(sender):
    """Extracts the domain from a sender string."""
    match = re.search(r'<.*@(.*?)>', sender)
    if match:
        return match.group(1)
    if '@' in sender:
        return sender.split('@')[-1]
    return 'unknown_domain'

def has_undisclosed_recipients(receiver):
    """Checks for signs of a mass email campaign."""
    receiver_lower = str(receiver).lower()
    return 1 if 'undisclosed' in receiver_lower or receiver_lower == '' else 0

def get_domain_lexical_features(domain):
    """Creates numerical features based on the domain string itself."""
    features = {}
    features['domain_hyphen_count'] = domain.count('-')
    brands = ['paypal', 'amazon', 'google', 'apple', 'microsoft', 'ebay', 'netflix', 'facebook', 'instagram', 'bank']
    features['domain_contains_brand'] = 1 if any(brand in domain.lower() for brand in brands) else 0
    features['domain_length'] = len(domain)
    return pd.Series(features)

def find_first_url(text):
    """Finds the first http or https URL in a block of text."""
    match = re.search(r'https?://[^\s<>"]+|www\.[^\s<>"]+', str(text))
    if match:
        return match.group(0)
    return ''

# --- Custom Transformer Class Definition (needed for loading the model) ---
# In Email_phishing.py (and detector.py)
class URLFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_ = None

    def fit(self, X, y=None):
        # In fit, we determine the feature names from the first row
        # This makes sure the names are always correct
        test_url = find_first_url(X['body'].iloc[0])
        url_features_series = extract_url_features(test_url)
        self.feature_names_ = url_features_series.index.tolist()
        return self
        
    def transform(self, X, y=None):
        urls = X['body'].apply(find_first_url)
        url_features_df = urls.apply(extract_url_features)
        return url_features_df.to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)

# ==============================================================================
# MAIN DETECTOR LOGIC
# ==============================================================================

MODEL_PATH = 'hybrid_phishing_detector.pkl'

def load_model(path):
    """Loads the pre-trained model pipeline from disk."""
    try:
        model_pipeline = joblib.load(path)
        print("Model loaded successfully!")
        return model_pipeline
    except FileNotFoundError:
        print(f"Error: Model file not found at '{path}'.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        sys.exit(1)

def get_prediction_scores(model_pipeline, sender, subject, body):
    """
    Takes email components, engineers all features, and returns prediction scores.
    """
    input_data = {'sender': [sender], 'receiver': [''], 'date': [''], 'subject': [subject], 'body': [body]}
    input_df = pd.DataFrame(input_data)

    input_df['full_text'] = input_df['subject'] + ' ' + input_df['body']
    input_df['sender_domain'] = input_df['sender'].apply(get_sender_domain)
    input_df['is_undisclosed_recipients'] = input_df['receiver'].apply(has_undisclosed_recipients)
    
    lexical_features = input_df['sender_domain'].apply(get_domain_lexical_features)
    input_df = pd.concat([input_df, lexical_features], axis=1)

    probabilities = model_pipeline.predict_proba(input_df)

    prob_legitimate = probabilities[0][0]
    prob_phishing = probabilities[0][1]

    return prob_legitimate, prob_phishing

def display_results(legit_score, phish_score):
    """Formats and displays the prediction results."""
    print("\n" + "="*40)
    print("        DETECTION RESULTS")
    print("="*40)
    
    legit_percent = legit_score * 100
    phish_percent = phish_score * 100

    print(f"\nConfidence Score (Legitimate): {legit_percent:.2f}%")
    print(f"Confidence Score (Phishing):   {phish_percent:.2f}%")

    print("\n--- FINAL VERDICT ---")
    if phish_percent > 80:
        print("HIGH RISK: This email is almost certainly a phishing attempt.")
    elif phish_percent > 50:
        print("SUSPICIOUS: This email has several characteristics of a phishing attack. Proceed with caution.")
    elif phish_percent > 20:
        print("LIKELY SAFE: This email shows some minor suspicious signs, but is likely legitimate.")
    else:
        print("ALMOST CERTAINLY SAFE: This email appears to be legitimate.")
    print("="*40)

def main():
    """The main function to run the CLI tool."""
    print("--- Phishing Email Detector ---")
    
    pipeline = load_model(MODEL_PATH)

    print("\nPlease provide the email details below:")
    sender_input = input("Enter Sender's Email Address/Name: ")
    subject_input = input("Enter Email Subject: ")
    
    print("\nPaste the email body below. When you are finished:")
    print("- On Linux/macOS: Press Ctrl+D on a new line.")
    print("- On Windows: Press Ctrl+Z followed by Enter on a new line.")
    body_lines = sys.stdin.readlines()
    body_input = "".join(body_lines)

    if not body_input.strip():
        print("\nError: Email body cannot be empty.")
        return

    legit_score, phish_score = get_prediction_scores(pipeline, sender_input, subject_input, body_input)
    display_results(legit_score, phish_score)

if __name__ == "__main__":
    main()