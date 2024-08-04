import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Donner le chemin vers dataset de train (biological, cellular).
path_train = '/home/sudo_dev/Desktop/Challenges/LLMs4OL/model/tools/cutedJson/molecular/molecular_1.json'

# Charger les données depuis un fichier JSON
with open(path_train, 'r') as f:
    data = json.load(f)

terms = [item['term'] for item in data]
types = [item['type'] for item in data]

# Encoder les types comme labels binaires (multi-hot encoding)
unique_types = list(set([t for sublist in types for t in sublist]))
type_to_id = {t: i for i, t in enumerate(unique_types)}
labels = [[type_to_id[t] for t in sublist] for sublist in types]

# Créer des vecteurs multi-hot pour les labels
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer(classes=unique_types)
labels = mlb.fit_transform(types)

# Charger le tokenizer et le modèle BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(unique_types))

# Tokenizer les termes
inputs = tokenizer(terms, padding=True, truncation=True, return_tensors="pt")

# Split les données en train et test
train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs['input_ids'], labels, test_size=0.2, random_state=42)

# Créer un dataset PyTorch
class TermDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {'input_ids': self.inputs[idx], 'labels': torch.tensor(self.labels[idx], dtype=torch.float)}

train_dataset = TermDataset(train_inputs, train_labels)
test_dataset = TermDataset(test_inputs, test_labels)

# Configuration de l'entraînement
# training_args = TrainingArguments(
#     output_dir='./results_molecular',
#     num_train_epochs=3,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     logging_steps=10,
# )
training_args = TrainingArguments(
    output_dir='./results_molecular',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50, 
    save_strategy="epoch", 
    save_total_limit=1,  
    load_best_model_at_end=True,  
    evaluation_strategy="epoch",  
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Entraîner le modèle
trainer.train()

# Sauvegarder le modèle et le tokenizer
model.save_pretrained('./saved_molecular_model')
tokenizer.save_pretrained('./saved_molecular_model_tokenizer')

# Évaluer le modèle
trainer.evaluate()

