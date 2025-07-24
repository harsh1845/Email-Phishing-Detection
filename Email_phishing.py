import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ==============================================================================
# 1. DATA LOADING AND UNIFICATION
# ==============================================================================
print("--- 1. Loading and Unifying All Datasets ---")

def load_and_standardize(filepath, column_map):
    """Loads a CSV and renames columns to a standard format."""
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip')
        df.rename(columns=column_map, inplace=True)
        # Ensure all standard columns exist, filling with None if not
        for col in ['sender', 'receiver', 'date', 'subject', 'body', 'label']:
            if col not in df.columns:
                df[col] = None
        return df[['sender', 'receiver', 'date', 'subject', 'body', 'label']]
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Skipping.")
        return pd.DataFrame()

# Define the column mappings for each file
enron_map = {'subject': 'subject', 'body': 'body', 'label': 'label'}
ling_map = {'subject': 'subject', 'body': 'body', 'label': 'label'}
ceas_map = {'sender': 'sender', 'receiver': 'receiver', 'date': 'date', 
            'subject': 'subject', 'body': 'body', 'label': 'label'}
spamassasin_map = {'sender': 'sender', 'receiver': 'receiver', 'date': 'date', 
                   'subject': 'subject', 'body': 'body', 'label': 'label'}

# Load and standardize all datasets
df_enron = load_and_standardize('Enron.csv', enron_map)
df_ling = load_and_standardize('Ling.csv', ling_map)
df_ceas = load_and_standardize('CEAS_08.csv', ceas_map)
df_spamassasin = load_and_standardize('SpamAssasin.csv', spamassasin_map)

# Combine into a single master DataFrame
df_master = pd.concat([df_enron, df_ling, df_ceas, df_spamassasin], ignore_index=True)

# --- Data Cleaning ---
# Drop rows where the label is missing
df_master.dropna(subset=['label'], inplace=True)
# Fill missing text/header data with empty strings
df_master.fillna('', inplace=True)
# Ensure label is integer
df_master['label'] = df_master['label'].astype(int)

print(f"Total emails in master dataset: {len(df_master)}")
print("Label distribution:")
print(df_master['label'].value_counts())

# ==============================================================================
# 2. HYBRID FEATURE ENGINEERING
# ==============================================================================
print("\n--- 2. Engineering Text and Header Features ---")

# --- A. Text Feature Preparation ---
# Combine subject and body for NLP analysis
df_master['full_text'] = df_master['subject'] + ' ' + df_master['body']

# --- B. Header Feature Engineering Functions ---
def get_sender_domain(sender):
    """Extracts the domain from a sender string like 'Name <email@domain.com>'"""
    match = re.search(r'<.*@(.*?)>', sender)
    if match:
        return match.group(1)
    if '@' in sender:
        return sender.split('@')[-1]
    return 'unknown_domain'

def has_undisclosed_recipients(receiver):
    """Checks for signs of a mass email campaign."""
    receiver_lower = receiver.lower()
    return 1 if 'undisclosed' in receiver_lower or receiver_lower == '' else 0

# Apply the feature engineering functions
df_master['sender_domain'] = df_master['sender'].apply(get_sender_domain)
df_master['is_undisclosed_recipients'] = df_master['receiver'].apply(has_undisclosed_recipients)

print("New header features created ('sender_domain', 'is_undisclosed_recipients').")
print(df_master[['sender', 'sender_domain', 'receiver', 'is_undisclosed_recipients']].head())

# ==============================================================================
# 3. BUILDING THE HYBRID PREPROCESSING PIPELINE
# ==============================================================================
print("\n--- 3. Building a Hybrid Preprocessing Pipeline with ColumnTransformer ---")

# --- NLP Preprocessing Function for the text pipeline ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(lemmas)

# --- Define the Text Pipeline ---
# This pipeline will take the 'full_text' column, preprocess it, and apply TF-IDF
text_pipeline = Pipeline([
    # This custom preprocessing step is no longer needed here, as TfidfVectorizer has built-in options
    ('tfidf', TfidfVectorizer(preprocessor=preprocess_text, max_features=5000, ngram_range=(1,2)))
])

# --- Define the Header Pipeline ---
# This pipeline will take the 'sender_domain' column and One-Hot Encode it.
# One-Hot Encoding turns categorical variables into a numerical format.
header_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=100)) # handle_unknown ignores domains not seen in training
])

# --- Create the ColumnTransformer to Combine Pipelines ---
# This is the core of the hybrid approach. It applies different pipelines to different columns.
preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_pipeline, 'full_text'),
        ('headers', header_pipeline, ['sender_domain']),
        # We can pass through numerical features directly
        ('pass_numeric', 'passthrough', ['is_undisclosed_recipients'])
    ],
    remainder='drop' # Drop any columns we haven't specified
)

# ==============================================================================
# 4. TRAINING AND EVALUATING THE HYBRID MODEL
# ==============================================================================
print("\n--- 4. Training and Evaluating the Hybrid Model ---")

# Define our features (the full DataFrame) and target
X = df_master
y = df_master['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create the full model pipeline including the preprocessor and the classifier
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', random_state=42))
])

# Train the entire pipeline
print("Training the full hybrid pipeline...")
full_pipeline.fit(X_train, y_train)
print("Training complete.")

# Make predictions
y_pred = full_pipeline.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\nHybrid Model Accuracy: {accuracy * 100:.2f}%")
print("\nHybrid Model Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate (0)', 'Phishing (1)']))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Hybrid Model Confusion Matrix')
plt.show()

# ==============================================================================
# 5. ANALYZING HYBRID FEATURE IMPORTANCE
# ==============================================================================
print("\n--- 5. Analyzing Hybrid Feature Importance ---")

# Extract the trained classifier and preprocessor from the pipeline
classifier = full_pipeline.named_steps['classifier']
preprocessor_fitted = full_pipeline.named_steps['preprocessor']

# Get feature names from all parts of the ColumnTransformer
text_features = preprocessor_fitted.named_transformers_['text']['tfidf'].get_feature_names_out()
header_features = preprocessor_fitted.named_transformers_['headers']['onehot'].get_feature_names_out(['sender_domain'])
numeric_features = ['is_undisclosed_recipients']

# Combine all feature names in the correct order
all_feature_names = np.concatenate([text_features, header_features, numeric_features])

# Get coefficients and create DataFrame
coefficients = classifier.coef_[0]
coef_df = pd.DataFrame({'feature': all_feature_names, 'coefficient': coefficients})

# Show top phishing indicators (text, headers, or numeric)
print("\nTop 20 Indicators for PHISHING (from all features):")
print(coef_df.sort_values(by='coefficient', ascending=False).head(20))

# Show top legitimate indicators
print("\nTop 20 Indicators for LEGITIMATE (from all features):")
print(coef_df.sort_values(by='coefficient', ascending=True).head(20))

print("\n--- Saving the algorithm in hybrid_phishing_detector.pkl")
joblib.dump(full_pipeline, 'hybrid_phishing_detector.pkl')

print("\n--- Project run complete ---")