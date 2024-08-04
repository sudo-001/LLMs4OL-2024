import pandas as pd
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import joblib

# Fonction pour charger les données
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

# Fonction pour charger les types
def load_types(file_path):
    with open(file_path, 'r') as f:
        types = f.read().splitlines()
    return types

# Chemins d'accès aux fichiers
geonames_train_file = '/home/sudo_dev/Desktop/Challenges/LLMs4OL/model/Datasets/TaskB-Taxonomy Discovery/SubTaskB.1-GeoNames/geoname_train_pairs.json'
go_train_file = '/home/sudo_dev/Desktop/Challenges/LLMs4OL/model/Datasets/TaskB-Taxonomy Discovery/SubTaskB.4-GO/go_train_pairs.json'
schema_train_file = '/home/sudo_dev/Desktop/Challenges/LLMs4OL/model/Datasets/TaskB-Taxonomy Discovery/SubTaskB.2-Schema.org/schemaorg_train_pairs.json'

# Charger les données
geonames_data = load_data(geonames_train_file)
go_data = load_data(go_train_file)
schema_data = load_data(schema_train_file)

# Charger les types
geonames_types = load_types('/home/sudo_dev/Desktop/Challenges/LLMs4OL/model/Datasets/TaskB-Taxonomy Discovery/SubTaskB.1-GeoNames/geoname_train_types.txt')
go_types = load_types('/home/sudo_dev/Desktop/Challenges/LLMs4OL/model/Datasets/TaskB-Taxonomy Discovery/SubTaskB.4-GO/go_train_types.txt')
schema_types = load_types('/home/sudo_dev/Desktop/Challenges/LLMs4OL/model/Datasets/TaskB-Taxonomy Discovery/SubTaskB.2-Schema.org/schemaorg_train_types.txt')

# Prétraitement
def preprocess(text):
    return text.lower().strip()

geonames_data['parent'] = geonames_data['parent'].apply(preprocess)
geonames_data['child'] = geonames_data['child'].apply(preprocess)
go_data['parent'] = go_data['parent'].apply(preprocess)
go_data['child'] = go_data['child'].apply(preprocess)
schema_data['parent'] = schema_data['parent'].apply(preprocess)
schema_data['child'] = schema_data['child'].apply(preprocess)

# Tokenizer et modèle BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()

# Fonction pour ajouter les types aux données
def add_type_features(df, types_list):
    type_dict = {type_: idx for idx, type_ in enumerate(types_list)}
    df['parent_type'] = df['parent'].apply(lambda x: type_dict.get(x, -1))
    df['child_type'] = df['child'].apply(lambda x: type_dict.get(x, -1))
    return df

geonames_data = add_type_features(geonames_data, geonames_types)
go_data = add_type_features(go_data, go_types)
schema_data = add_type_features(schema_data, schema_types)

# Extraction des embeddings
geonames_data['parent_emb'] = geonames_data['parent'].apply(get_embeddings)
geonames_data['child_emb'] = geonames_data['child'].apply(get_embeddings)
go_data['parent_emb'] = go_data['parent'].apply(get_embeddings)
go_data['child_emb'] = go_data['child'].apply(get_embeddings)
schema_data['parent_emb'] = schema_data['parent'].apply(get_embeddings)
schema_data['child_emb'] = schema_data['child'].apply(get_embeddings)

# Combiner les caractéristiques et aplatir les embeddings
def combine_features(df):
    X = []
    for parent_emb, child_emb, parent_type, child_type in zip(df['parent_emb'].values, df['child_emb'].values, df['parent_type'].values, df['child_type'].values):
        combined = np.hstack((parent_emb, child_emb, [parent_type, child_type]))  # Combiner et aplatir les embeddings avec les types
        X.append(combined)
    return np.array(X)

geonames_X = combine_features(geonames_data)
go_X = combine_features(go_data)
schema_X = combine_features(schema_data)

# Création des étiquettes (exemple avec étiquettes 0 et 1 alternées)
geonames_y = [0 if i % 2 == 0 else 1 for i in range(len(geonames_X))]
go_y = [0 if i % 2 == 0 else 1 for i in range(len(go_X))]
schema_y = [0 if i % 2 == 0 else 1 for i in range(len(schema_X))]

# Combiner toutes les données
X_combined = np.vstack((geonames_X, go_X, schema_X))
y_combined = np.hstack((geonames_y, go_y, schema_y))

# Vérifiez la taille des données
print("Taille des données combinées:")
print("X_combined:", X_combined.shape)
print("y_combined:", len(y_combined))

# Validation croisée
print("Validation croisée en cours...")
clf = RandomForestClassifier()
scores = cross_val_score(clf, X_combined, y_combined, cv=5)
print("Validation croisée - Scores:", scores)
print("Validation croisée - Précision moyenne:", scores.mean())

# Division en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Entraînement du modèle
print("Entraînement du modèle...")
clf.fit(X_train, y_train)
print("Modèle entraîné.")

# Évaluation du modèle
print("Évaluation du modèle...")
y_pred = clf.predict(X_val)
precision = precision_score(y_val, y_pred, average='macro')
recall = recall_score(y_val, y_pred, average='macro')
f1 = f1_score(y_val, y_pred, average='macro')
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Sauvegarder le modèle
joblib.dump(clf, 'combined_taskB_model.joblib')
