import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib
import pandas as pd
import re
import nltk
import zipfile

# Charger le modèle, le tokenizer et l'encodeur de labels
model = BertForSequenceClassification.from_pretrained('./saved_model')
tokenizer = BertTokenizer.from_pretrained('./saved_tokenizer')
label_encoder = joblib.load('label_encoder.joblib')

# Assurez-vous que le modèle est en mode évaluation
model.eval()

# Télécharger les ressources NLTK nécessaires
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Définir les stop words et le lemmatizer
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r'\W', ' ', text)
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Charger et prétraiter les données de test
# test_file_path = './Train dataset/A.2(FS)_GeoNames_Test.json'
# test_file_path = './Train dataset/A.4(FS)_GO_Biological_Process_Test.json'
# test_file_path = './Train dataset/A.4(FS)_GO_Cellular_Component_Test.json'
# test_file_path = './Train dataset/A.4(FS)_GO_Molecular_Function_Test.json'
test_file_path = '/home/sudo_dev/Desktop/Challenges/LLMs4OL/model/Train dataset/A.5(ZS)_Test.json'

with open(test_file_path, 'r') as f:
    test_data = json.load(f)

test_df = pd.DataFrame(test_data)

# Vérifier les colonnes disponibles
print(test_df.columns)  # Ajoutez cette ligne pour voir quelles colonnes sont présentes

# Utiliser la colonne correcte pour les phrases
if 'sentence' in test_df.columns:
    test_df['processed_sentence'] = test_df['sentence'].fillna('').apply(preprocess)
elif 'term' in test_df.columns:  # Si 'sentence' n'existe pas, utilisez 'term'
    test_df['processed_sentence'] = test_df['term'].fillna('').apply(preprocess)
else:
    raise KeyError("Neither 'sentence' nor 'term' column found in the test data")

# Tokenization des données de test par lots
batch_size = 16
predicted_labels = []

for i in range(0, len(test_df), batch_size):
    batch_sentences = list(test_df['processed_sentence'][i:i+batch_size])
    batch_ids = list(test_df['ID'][i:i+batch_size])
    test_encodings = tokenizer(batch_sentences, truncation=True, padding=True, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**test_encodings)
        logits = outputs.logits
        predictions = torch.sigmoid(logits).cpu().numpy()

    # Décoder les prédictions
    threshold = 0.5  # Vous pouvez ajuster ce seuil selon vos besoins
    for prediction in predictions:
        predicted_indexes = (prediction > threshold).nonzero()[0]
        predicted_types = label_encoder.inverse_transform(predicted_indexes)
        predicted_labels.append(predicted_types.tolist())

# Créer la liste de dictionnaires selon le format requis
submission_data = []
for i, row in test_df.iterrows():
    sample_id = row['ID']
    types = predicted_labels[i]
    submission_data.append({"ID": sample_id, "type": types})

# Sauvegarder le fichier JSON
with open('A.5(ZS)_Test.json', 'w') as f:
    json.dump(submission_data, f, indent=4)

# Compresser le fichier JSON
with zipfile.ZipFile('A.5(ZS)_Test.zip', 'w') as zipf:
    zipf.write('A.5(ZS)_Test.json')

print("Fichier de soumission enregistré avec succès : A.5(ZS)_Test.zip")


