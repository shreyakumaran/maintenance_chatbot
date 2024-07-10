from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from transformers import pipeline
import spacy

app = Flask(__name__)
CORS(app)

# Load data
data = pd.read_csv('MSR_REQ_202407091429.csv')
features = data.columns.tolist()
features_lower = [feature.lower() for feature in features]

# Load NLP models
nlp = spacy.load('en_core_web_sm')
qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

def extract_entities(text):
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities

@app.route('/query', methods=['POST'])
def handle_query():
    user_query = request.json.get('query')
    
    preprocessed_query = preprocess_text(user_query)
    entities = extract_entities(preprocessed_query)
    
    # Identify the field
    field = None
    for feature in features_lower:
        if feature in user_query.lower():
            field = feature
            break
    
    if not field:
        return jsonify({"error": "Field not found in the database"}), 400
    
    # Determine the machine component
    component = None
    for entity in entities:
        if entity[1] == 'ORG':  # Assuming machine components are tagged as ORG in NER
            component = entity[0]
            break
    
    if not component:
        return jsonify({"error": "Component not found"}), 400
    
    # Retrieve the answer
    result = data[data['machine_component'].str.lower() == component.lower()][field].values
    if len(result) == 0:
        return jsonify({"error": "No records found"}), 404
    
    return jsonify({"result": result[0]})

if __name__ == '__main__':
    app.run(debug=True)
