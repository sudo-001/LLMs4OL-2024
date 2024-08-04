import pandas as pd
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from transformers import BertTokenizer, BertModel
import torch
import joblib

# Charger le modèle entraîné
clf = joblib.load('combined_model.joblib')

# Fonction pour charger les types à partir d'un fichier .txt
def load_types(file_path):
    with open(file_path, 'r') as f:
        types = f.read().splitlines()
    return types

# Prétraitement
def preprocess(text):
    return text.lower().strip()

# Extraction de caractéristiques avec BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Préparer les données pour la prédiction
def create_pairs(types):
    pairs = []
    for i, parent in enumerate(types):
        for j, child in enumerate(types):
            if i != j:  # éviter les paires parent-enfant identiques
                pairs.append((parent, child))
    return pairs

# Combiner les caractéristiques des paires pour la prédiction
def combine_features(pairs, tokenizer, model):
    X = []
    for parent, child in pairs:
        parent_emb = get_embeddings(parent, tokenizer, model)
        child_emb = get_embeddings(child, tokenizer, model)
        combined = np.hstack((parent_emb.flatten(), child_emb.flatten()))  # Combiner et aplatir les embeddings
        X.append(combined)
    return np.array(X)

# Fonction pour prédire les relations et générer le fichier JSON
def predict_and_generate_json(model, pairs, tokenizer, bert_model, output_file):
    test_X = combine_features(pairs, tokenizer, bert_model)
    y_pred = model.predict(test_X)
    predictions = [
        {"parent": pairs[idx][0], "child": pairs[idx][1]}
        for idx, pred in enumerate(y_pred) if pred == 1
    ]
    
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=4)

# Chemin d'accès au fichier de test et de sortie
test_file = '/home/sudo_dev/Desktop/Challenges/LLMs4OL/model/Test dataset/TaskB/B.2(FS)_Schemaorg_Test.txt'
output_file = '/home/sudo_dev/Desktop/Challenges/LLMs4OL/model/submissions_taskB/B.2(FS)_Schemaorg_Test.json'

# Charger et prétraiter les types
types = load_types(test_file)
types = [preprocess(t) for t in types]

# Créer des paires parent-enfant
pairs = create_pairs(types)

# Prédire les relations et générer le fichier JSON
predict_and_generate_json(clf, pairs, tokenizer, model, output_file)

print(f"Les prédictions ont été sauvegardées dans {output_file}")
