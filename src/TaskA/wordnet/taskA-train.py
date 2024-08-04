import json
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import nltk
import joblib


# Telecharger les ressources NLTK necessaires
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Definir les stop words et le lemmatizer
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r'\W', ' ', text)
    tokens = nltk.word_tokenize(text.lower())
    # tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def load_and_preprocess_wordnet(file_path):
    with open(file_path, 'r') as f :
        data = json.load(f)
    df = pd.DataFrame(data)
    df['sentence'] = df['sentence'].fillna('')
    df['processed_sentence'] = df['sentence'].fillna('').apply(preprocess)
    return df

# Charger les donnes Wordnet
wordnet_train_file_path = './Datasets/TaskA-Term Typing/SubTaskA.1-WordNet/wordnet_train.json'
wordnet_train_df = load_and_preprocess_wordnet(wordnet_train_file_path)

# Tokinzation et encodage des etiquettes
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
label_encoder = LabelEncoder()
wordnet_train_df['encoded_type'] = label_encoder.fit_transform(wordnet_train_df['type'])

# Preparation des donnees pour BERT
X = list(wordnet_train_df['processed_sentence'])
y = list(wordnet_train_df['encoded_type'])

# Split des donnees en ensembles d'entrainement et de validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenization
train_encodings = tokenizer(X_train, truncation=True, padding=True)
val_encodings = tokenizer(X_val, truncation=True, padding=True)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)
    
train_dataset = Dataset(train_encodings, y_train)
val_dataset = Dataset(val_encodings, y_val)

# Chargement du modele
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

# Entrainement du modele
training_args = TrainingArguments(
    output_dir = './results',
    # num_train_epochs=3,
    num_train_epochs=5, # Augmenter le nombre d'epochs
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    # warmup_steps=500,
    warmup_steps = 1000, # Augmenter les etapes de warmup
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=2e-5 # Tester avec un taux d'apprentissage different
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# Evaluation du modele
eval_results = trainer.evaluate()

print("Evaluation Results : ", eval_results)

# Sauvegarde du modele
model.save_pretrained('./saved_model')

# Sauvegarder le tokenizer
tokenizer.save_pretrained('./saved_tokenizer')

# Sauvegarder l'encodeur de labels
joblib.dump(label_encoder, 'label_encoder.joblib')

# print("Accuracy : ", eval_results["eval_accuracy"])
# print("Precision : ", eval_results["eval_precision"])
# print("Recall : ", eval_results["eval_recall"])
# print("F1 Score : ", eval_results["eval_f1"])

