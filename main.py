import pandas as pd

# Load data
data = pd.read_csv('MSR_REQ_202407091429.csv')

# Extract columns dynamically
features = data.columns.tolist()
#print("Features:", features)
features_lower = [feature.lower() for feature in features]
from transformers import pipeline
import spacy

# Load SpaCy model for NER
nlp = spacy.load("en_core_web_trf")
import en_core_web_trf
nlp = en_core_web_trf.load()

# Load Hugging Face QA model
qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

# Function to preprocess text
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

# Function to extract entities
def extract_entities(text):
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities

def handle_query(query):
    preprocessed_query = preprocess_text(query)
    entities = extract_entities(preprocessed_query)
    
    # Identify the field by checking the presence of feature names in the query
    field = None
    for feature in features_lower:
        if feature in query.lower():
            field = feature
            break
    
    if not field:
        return "Field not found in the database"
    
    # Determine the machine component from the entities
    component = None
    for entity in entities:
        if entity[1] == 'ORG':  # Assuming machine components are tagged as ORG in NER
            component = entity[0]
            break
    
    if not component:
        return "Component not found"
    
    # Retrieve the answer
    result = data[data['machine_component'].str.lower() == component.lower()][field].values
    if len(result) == 0:
        return "No records found"
    
    return result[0]
