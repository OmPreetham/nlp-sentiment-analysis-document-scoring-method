import pandas as pd
import numpy as np
import nltk
import re
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm as sparse_norm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower().strip())
    
    # Tokenization and lemmatization
    words = nltk.word_tokenize(text)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word, pos='v') for word in words]  # Verbs are converted to base form
    
    return ' '.join(lemmatized_words)

# Function to preprocess labels
def preprocess_labels(labels):
    label_mapping = {
        'positive': 0,
        'negative': 1,
        'neutral': 2
    }
    return [label_mapping.get(label, label) for label in labels]

# Function to calculate TF-IDF scores according to project formulas
def calculate_tfidf(documents):
    # Vectorize documents
    vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize, token_pattern=None)  # Set token_pattern to None explicitly
    count_matrix = vectorizer.fit_transform(documents)
    
    # Check if the number of features is at least 200
    if len(vectorizer.get_feature_names_out()) < 200:
        raise ValueError("The number of features is less than 200. Consider reducing text preprocessing or using a larger corpus.")
    print(f"Number of features: {len(vectorizer.get_feature_names_out())}")
    
    # Compute term frequencies (TF)
    tf = count_matrix.astype(float)
    tf.data = np.log10(tf.data + 1)
    
    # Compute inverse document frequencies (IDF)
    df = np.log10(tf.shape[0] / (count_matrix.getnnz(axis=0) + 1))
    idf = np.array(df).flatten()
    
    # Compute TF-IDF scores
    tfidf = tf.multiply(idf)
    return csr_matrix(tfidf), vectorizer

# Function to train the TF-IDF model
def train_tfidf_model(train_documents, train_labels, test_documents, test_labels):
    tfidf_matrix, vectorizer = calculate_tfidf(train_documents)
    
    # Preprocess labels
    train_labels = preprocess_labels(train_labels)
    test_labels = preprocess_labels(test_labels)
    
    # Combine training and test labels, then fit the LabelEncoder
    all_labels = np.concatenate([train_labels, test_labels])
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(all_labels)
    
    # Get indices for training labels
    train_label_indices = encoded_labels[:len(train_labels)]
    
    # Compute average TF-IDF vector for each category
    tfidf_means = []
    for category in np.unique(train_label_indices):
        category_indices = (train_label_indices == category)
        category_tfidf = tfidf_matrix[category_indices].mean(axis=0)
        tfidf_means.append(category_tfidf)  # Keep as sparse matrix
    
    # Store model components
    model_components = {
        'vectorizer': vectorizer,
        'label_encoder': label_encoder,
        'tfidf_means': tfidf_means
    }
    dump(model_components, 'tfidf_model.pkl')

# Function to calculate the score for a document vector against a category TF-IDF vector
def calculate_score(doc_vector, category_vector):
    # Normalize the category TF-IDF vector
    if isinstance(category_vector, np.ndarray):
        category_vector_norm = np.linalg.norm(category_vector)
    else:  # It is sparse
        category_vector_norm = sparse_norm(category_vector)

    if category_vector_norm == 0:
        return 0
    # Calculate the score as per the provided formula
    score = (doc_vector * category_vector.T) / category_vector_norm
    return score if isinstance(score, float) else score[0, 0]

# Function to test the TF-IDF model using the scoring formula provided
def test_tfidf_model(test_documents, true_labels):
    model_components = load('tfidf_model.pkl')
    vectorizer = model_components['vectorizer']
    label_encoder = model_components['label_encoder']
    tfidf_means = model_components['tfidf_means']
    
    # Transform test documents
    X_test = vectorizer.transform(test_documents)
    X_test.data = np.log10(X_test.data + 1)  # Apply log base 10 for test TF
    
    predictions = []
    for doc_vector in X_test:
        # Score test document against each category's TF-IDF vector
        scores = [calculate_score(doc_vector, mean_vector) for mean_vector in tfidf_means]
        predicted_category = np.argmax(scores)
        predictions.append(predicted_category)
    
    # Decode labels and calculate accuracy, confusion matrix, and classification report
    predicted_labels = label_encoder.inverse_transform(predictions)
    true_labels = preprocess_labels(true_labels)  # Preprocess true labels
    accuracy = np.mean([y_hat == y_true for y_hat, y_true in zip(predicted_labels, true_labels)])
    print('Classification Report:')
    print(classification_report(true_labels, predicted_labels, target_names=['positive', 'negative', 'neutral']))
    print(f'Accuracy: {accuracy}')
    print('Contingency Table:')
    print(pd.DataFrame(confusion_matrix(true_labels, predicted_labels), index=['positive', 'negative', 'neutral'], columns=['positive', 'negative', 'neutral']))
    
try:
    train_data = pd.read_csv('trainData.csv')
    test_data = pd.read_csv('testData.csv')
    train_data['preprocessed_text'] = train_data['Description'].apply(preprocess_text)
    test_data['preprocessed_text'] = test_data['Description'].apply(preprocess_text)
    
    train_tfidf_model(train_data['preprocessed_text'], train_data['Category'], test_data['preprocessed_text'], test_data['Category'])
    test_tfidf_model(test_data['preprocessed_text'], test_data['Category'])
except Exception as e:
    print(f"An error occurred: {e}")
