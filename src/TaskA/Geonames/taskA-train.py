import json
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

# Charger les données
with open('geonames.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Préparation des données
terms = [item['term'] for item in data]
labels = [item['type'] for item in data]

# Encodeur de labels
label_to_id = {label: i for i, label in enumerate(set(labels))}
id_to_label = {i: label for label, i in label_to_id.items()}
encoded_labels = [label_to_id[label] for label in labels]

# Split des données en train et test
train_terms, test_terms, train_labels, test_labels = train_test_split(terms, encoded_labels, test_size=0.2, random_state=42)

# Tokenizer BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Dataset personnalisé
class GeoNamesDataset(Dataset):
    def __init__(self, terms, labels, tokenizer, max_len):
        self.terms = terms
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.terms)
    
    def __getitem__(self, idx):
        term = self.terms[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            term,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'term_text': term,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Paramètres
BATCH_SIZE = 16
MAX_LEN = 32
EPOCHS = 4
LEARNING_RATE = 2e-5

# DataLoader
train_dataset = GeoNamesDataset(train_terms, train_labels, tokenizer, MAX_LEN)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = GeoNamesDataset(test_terms, test_labels, tokenizer, MAX_LEN)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Modèle BERT
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_to_id))
model = model.to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

# Optimiseur et scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Fonction de perte
loss_fn = torch.nn.CrossEntropyLoss().to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

# Fonction de formation
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        logits = outputs[1]
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return correct_predictions.double() / n_examples, np.mean(losses)

# Fonction d'évaluation
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            logits = outputs[1]
            
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
    
    return correct_predictions.double() / n_examples, np.mean(losses)

# Entraînement du modèle
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    
    train_acc, train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer, device, scheduler, len(train_dataset))
    print(f'Train loss {train_loss} accuracy {train_acc}')
    
    val_acc, val_loss = eval_model(model, test_dataloader, loss_fn, device, len(test_dataset))
    print(f'Val   loss {val_loss} accuracy {val_acc}')

print("Modèle entraîné avec succès.")

# Sauvegarder le modèle
model.save_pretrained('bert-geonames-model')
tokenizer.save_pretrained('bert-geonames-tokenizer')
