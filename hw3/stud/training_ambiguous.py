import os
from transformers import (
    Trainer,
    BertTokenizer,
    BertModel
)
import transformers
import numpy as np
import pandas as pd

import random
import time
import yaml
from tqdm import tqdm
from typing import *
from datetime import datetime
from arguments import CustomTrainingArguments

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler

SEED = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# Display the entire text
pd.set_option("display.max_colwidth", None)


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_clean_path = "../../model/data/train_clean.tsv"
valid_clean_path = "../../model/data/valid_clean.tsv"

df_train = pd.read_csv(filepath_or_buffer=train_clean_path, sep="\t")
df_valid = pd.read_csv(filepath_or_buffer=valid_clean_path, sep="\t")

model_name_or_path = "bert-base-cased"

class GAP_AmbiguousDetection_Dataset(Dataset):
    """Custom GAP Dataset class"""

    def __init__(self, df, tokenizer, labeled=True):
        self.df = df

        self.labeled = labeled
        self.tokenizer = tokenizer
        self.tokens = []
        self.pronouns_offsets = []
        self.ambiguous_pron_offsets = []
#         self.original_offsets = []

        self._convert_tokens_to_ids()

        if labeled:
            self.labels = []
            
            self._assign_class_to_tokens()
        
            assert len(self.tokens) == len(self.labels)

        
    def _assign_class_to_tokens(self):        
        
        for sentence_idx, offsets_list in enumerate(self.pronouns_offsets):
            labels = []
            for offset in offsets_list:
                if offset == self.ambiguous_pron_offsets[sentence_idx]:
                    labels.append(2)
                else:
                    labels.append(1)
                    
            self.labels.append(labels)   
        
    def _convert_tokens_to_ids(self):
        CLS = [self.tokenizer.cls_token]
        SEP = [self.tokenizer.sep_token]

        for _, row in self.df.iterrows():
            tokens, ambiguous_pron_offset, pronouns_offset = self._tokenize(row)
            self.tokens.append(self.tokenizer.convert_tokens_to_ids(
                CLS + tokens + SEP))
            
            # Because of the introduction of CLS we have to add 1 to the offsets
            self.pronouns_offsets.append(pronouns_offset)

            if ambiguous_pron_offset:
                self.ambiguous_pron_offsets.append(ambiguous_pron_offset+1)
            
            
   

    def _insert_tag(self, text, pronoun_offset):
        """Insert custom tags to help us find the position of A, B, and the pronoun after tokenization."""
        to_be_inserted = sorted([
        #         (a_offset, " [A] "),
        #         (b_offset, " [B] "),
            (pronoun_offset, " <P> ")
        ], key=lambda x: x[0], reverse=True)

        for offset, tag in to_be_inserted:
            text = text[:offset] + tag + text[offset:]
        return text

    def _tokenize(self, row):
        """Returns a list of tokens and the positions of A, B, and the pronoun."""
        entries = {}
        final_tokens = []
        pronouns_offsets = []
        pronoun_list = ['he','she','him','her','his','hers']
        
        text = self._insert_tag(row['text'], row['p_offset'])
        for token in self.tokenizer.tokenize(text):
            if token == ("<P>"):
                entries[token] = len(final_tokens)
                continue
                
            if token.lower() in pronoun_list:
                pronouns_offsets.append(len(final_tokens)+1)
            
            final_tokens.append(token)
        return final_tokens, entries["<P>"], pronouns_offsets

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        if self.labeled:
            assert len(self.pronouns_offsets[idx]) == len(self.labels[idx])
            return self.tokens[idx], self.pronouns_offsets[idx], self.labels[idx]
        return self.tokens[idx], self.pronouns_offsets[idx], None


tokenizer_path = "../../model/tokenizer/"
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

train_ds = GAP_AmbiguousDetection_Dataset(df_train, tokenizer)
valid_ds = GAP_AmbiguousDetection_Dataset(df_valid, tokenizer)


class Collator:
    def __init__(self, device):
        self.device = device
        
    def __call__(self, batch, truncate_len=400):
        """Batch preparation.

        1. Pad the sequences
        2. Transform the target.
        """
        batch_features, batch_pronouns_offsets, batch_labels = zip(*batch)

        max_len_features_in_batch = self.compute_max_len(batch_features, truncate_len)
        max_len_offsets_in_batch = self.compute_max_len(batch_pronouns_offsets, truncate_len)

        # Features        
        padded_features = self.pad_sequence(batch_features, max_len_features_in_batch, 0)
        features_tensor = torch.tensor(padded_features, device=device)

        # Offsets
        padded_pronouns_offsets = self.pad_sequence(batch_pronouns_offsets, max_len_offsets_in_batch, 0)
        pronouns_offsets_tensor = torch.tensor(padded_pronouns_offsets, device=device)

        # Labels
        if batch_labels[0] is None:
            return features_tensor, pronouns_offsets_tensor, None

        padded_labels = self.pad_sequence(batch_labels, max_len_offsets_in_batch, 0)
        labels_tensor = torch.tensor(padded_labels, dtype=torch.uint8, device=device)

        return features_tensor, pronouns_offsets_tensor, labels_tensor
    
    @staticmethod
    def compute_max_len(sentences, truncate_len) -> int:
        # calculate the max sentence length in the dataset
        max_len = min(
            max((len(x) for x in sentences)),
            truncate_len
        )
        return max_len
    
    @staticmethod
    def pad_sequence(list_sequences: List[List[Any]], max_len: int, pad: int) -> List[Any]:
    
        features = np.full((len(list_sequences), max_len), pad, dtype=np.int64)

        # Padding
        for i, row in enumerate(list_sequences):
            features[i, :len(row)] = row

        return features


class CorefHead(nn.Module):
    def __init__(self, bert_hidden_size: int):
        super().__init__()
        self.bert_hidden_size = bert_hidden_size
        self.head_hidden_size = 512

        # a) Always BN -> AC, (Nothing b/w them).
        # b) BN -> Dropout over Dropout -> BN, but try both. [Newer research, finds 1st better ]
        # c) BN eliminates the need of Dropout, no need to use Dropout.
        # e) BN before Dropout is data Leakage.
        # f) Best thing is to try every combination.
        # SO CALLED BEST METHOD -
        # Layer -> BN -> AC -> Dropout ->Layer

        self.fc = nn.Sequential(
            #             nn.BatchNorm1d(bert_hidden_size * 3),
            #             nn.Dropout(0.5),
            #             nn.LeakyReLU(),
            #             nn.Linear(bert_hidden_size * 3, self.head_hidden_size),
            #             nn.BatchNorm1d(self.head_hidden_size),
            #             nn.Dropout(0.5),
            #             nn.Linear(self.head_hidden_size, self.head_hidden_size),
            #             nn.ReLU(),
            #             nn.BatchNorm1d(self.head_hidden_size),
            #             nn.Dropout(0.5),
#             nn.Dropout(0.1),
#             nn.Linear(bert_hidden_size * 3, self.head_hidden_size),
#             nn.BatchNorm1d(self.head_hidden_size),
#             nn.LeakyReLU(),
#             nn.Linear(self.head_hidden_size, 3)
            nn.Linear(bert_hidden_size, self.head_hidden_size),
            # nn.BatchNorm1d(self.head_hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
        )


        # lstm_hidden_dim = 256
        # bidirectional = True
        # self.lstm = nn.LSTM(self.head_hidden_size, 
        #                     lstm_hidden_dim, 
        #                     bidirectional=bidirectional,
        #                     num_layers=1,
        #                     dropout=0.1,
        #                     batch_first=True)
        
        # lstm_output_dim = lstm_hidden_dim if bidirectional is False \
        #                     else lstm_hidden_dim * 2

        # self.relu = nn.LeakyReLU()
        
        self.classifier = nn.Linear(self.head_hidden_size, 3)
        # self.classifier = nn.Linear(lstm_output_dim, 3)

    def forward(self, bert_outputs, offsets):
        embeddings = self._retrieve_pronouns_embeddings(bert_outputs, offsets)

        x = self.fc(embeddings)
        # x, _ = self.lstm(x)
        # x = self.relu(x)

        output = self.classifier(x)
        return output
    
    def _retrieve_pronouns_embeddings(self, bert_embeddings, pronouns_offsets):
        pronouns_embeddings = []

        # Consider embeddings and offsets in each batch separately
        for embeddings, offsets in zip(bert_embeddings, pronouns_offsets):
            pronouns_embeddings.append(embeddings[offsets])

        # Merge outputs
        merged_pronouns_embeddings = torch.stack(pronouns_embeddings, dim=0)
        
        # shape: batch_size x seq_length x embedding_dim
        return merged_pronouns_embeddings

class GAPModel(nn.Module):
    """The main model."""

    def __init__(self, bert_model: str):
        super().__init__()

        if bert_model in {"bert-base-uncased", "bert-base-cased"}:
            self.bert_hidden_size = 768
        elif bert_model in {"bert-large-uncased", "bert-large-cased"}:
            self.bert_hidden_size = 1024
        else:
            raise ValueError("Unsupported BERT model.")

        self.bert = BertModel.from_pretrained(
            bert_model).to(device, non_blocking=True)
        self.head = CorefHead(self.bert_hidden_size).to(
            device, non_blocking=True)

    def forward(self, x, offsets):
        bert_outputs = self.bert(
            x, attention_mask=(x > 0).long(),
            token_type_ids=None, output_hidden_states=True)
#         concat_bert = torch.cat((bert_outputs[-1],bert_outputs[-2],bert_outputs[-3]),dim=-1)
        # concat_bert = torch.cat((bert_outputs.hidden_states[-1], bert_outputs.hidden_states[-2],
        #                          bert_outputs.hidden_states[-3], bert_outputs.hidden_states[-4]), dim=-1)
        
        out = bert_outputs.last_hidden_state

        # layers_to_sum = torch.stack([bert_outputs.hidden_states[x] for x in [-1, -2, -3, -4]], dim=0)
        # out = torch.sum(layers_to_sum, dim=0)

        # out = torch.cat([bert_outputs.hidden_states[x] for x in [-1, -2, -3, -4]], dim=-1)

        head_outputs = self.head(out, offsets)
#         return concat_bert
        return head_outputs

class Trainer:
    
    def __init__(
        self,
        model: nn.Module,
        args: CustomTrainingArguments,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        criterion: torch.nn,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        
    ):
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        if args is None:
            output_dir = "../../model/tmp_trainer"
            print(f"No 'TrainingArguments' passed, using 'output_dir={output_dir}'.")
            args = CustomTrainingArguments(output_dir=output_dir)
        
        self.args = args
        
    def train(self):
        args = self.args
        valid_dataloader = self.valid_dataloader
        epochs = args.num_train_epochs
        
        train_losses = []
        train_acc_list = []
        valid_losses = []
        valid_acc_list = []
        
        if args.early_stopping:
            patience_counter = 0 

        scaler = GradScaler()

        if args.resume_from_checkpoint is not None:
            self._resume_model(args.resume_from_checkpoint, scaler)

        training_start_time = time.time()
        print("\nTraining...")
        for epoch in range(epochs):
            train_loss, train_acc = self._inner_training_loop(scaler)
            train_losses.append(train_loss)
            train_acc_list.append(train_acc)

            valid_loss, valid_acc = self.evaluate(valid_dataloader)
            valid_losses.append(valid_loss)
            valid_acc_list.append(valid_acc)

            if self.scheduler is not None:
                print('-' * 17)
                print(f"| LR: {self.scheduler.get_last_lr()[0]:.3e} |")
                self.scheduler.step()

            self._print_epoch_log(epoch, epochs, train_loss, valid_loss, valid_acc)

            if args.early_stopping and len(valid_acc_list) >= 2:
                # stop = args.early_stopping_mode == 'min' and epoch > 0 and valid_acc_list[-1] > valid_acc_list[-2]
                stop = args.early_stopping_mode == 'max' and epoch > 0 and valid_acc_list[-1] < valid_acc_list[-2]
                if stop:
                    if patience_counter >= args.early_stopping_patience:
                        print('Early stop.')
                        break
                    else:
                        print('-- Patience.\n')
                        patience_counter += 1
        
        training_time = time.time() - training_start_time
        print(f'Training time: {training_time:.2f}s')

        metrics_history = {
            "train_losses": train_losses,
            "train_acc": train_acc_list,
            "valid_losses": valid_losses,
            "valid_acc": valid_acc_list,
        }
        print(metrics_history)
        if args.save_model:
            self._save_model("1", epoch, valid_acc, scaler, metrics_history)
    
        return #metrics_history

    def _inner_training_loop(self, scaler):
        args = self.args
        train_dataloader = self.train_dataloader
        
        train_loss = 0.0
        train_correct, total_count = 0.0, 0.0

        self.model.train()
        for step, (features, offsets, labels) in enumerate(train_dataloader):
            # Empty gradients
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward
            with torch.cuda.amp.autocast(): # autocast as a context manager
                predictions = self.model(features, offsets)
                predictions = predictions.view(-1, predictions.shape[-1])
                labels = labels.view(-1)
                loss = self.criterion(predictions, labels)

            # predictions = predictions.view(-1, predictions.shape[-1])
            # labels = labels.view(-1)
            # loss = self.criterion(predictions, labels)

            mask = labels != 0
            predictions = predictions.argmax(1)
            predictions = predictions[mask]
            labels = labels[mask]
            train_correct += (predictions == labels).sum().item()
            total_count += labels.shape[0]
            
            # Backward  
            # loss.backward()
            # Backward pass without mixed precision
            # It's not recommended to use mixed precision for backward pass
            # Because we need more precise loss
            scaler.scale(loss).backward()
            
            if args.grad_clipping is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clipping)
            
            # Update weights 
            # self.optimizer.step()
            scaler.step(self.optimizer)
            scaler.update()

            train_loss += loss.item()

            if step % args.logging_steps == args.logging_steps - 1:
                running_loss = train_loss / (step + 1)
                running_acc = train_correct / total_count
                self._print_step_log(step, running_loss, running_acc)
                
        return train_loss / len(train_dataloader), train_correct / total_count


    def evaluate(self, eval_dataloader):
        valid_loss = 0.0
        eval_correct, total_count = 0, 0
        
        self.model.eval()
        with torch.no_grad():
            for (features, offsets, labels) in eval_dataloader:
                
                predictions = self.model(features, offsets)
                
                predictions = predictions.view(-1, predictions.shape[-1])
                labels = labels.view(-1)
                loss = self.criterion(predictions, labels)
                valid_loss += loss.item()
                
#                 accuracy = compute_score(predictions, labels)
                
                mask = labels != 0
                predictions = predictions.argmax(1)
                predictions = predictions[mask]
                labels = labels[mask]
                eval_correct += (predictions == labels).sum().item()
                total_count += labels.shape[0]
        
        return valid_loss / len(eval_dataloader), eval_correct / total_count

    
    def compute_score(self, predictions: torch.Tensor, labels: torch.Tensor):
        mask = labels != 0
        labels = labels[mask]        
        
        predictions = predictions[mask]
        maximum_logits, predicted_labels = predictions.max(1)
        
        # It may happen that more than one pronoun is classify as ambiguous
        multiple_ambiguous_pronouns_mask = predicted_labels == 2
        ambiguous_pronouns_logits = maximum_logits[multiple_ambiguous_pronouns_mask]

        # More than one pronoun is classify as ambiguous
        if len(ambiguous_pronouns_logits) > 1:
            # Get the highest logit among the ambiguous ones
            highest_ambiguous_pronoun_logit = ambiguous_pronouns_logits.max()

            # Identity the position of the logit that should correspond to the ambiguous prononun class (2)
            ambiguous_pronoun_mask = maximum_logits == highest_ambiguous_pronoun_logit

            # All the predictions that are not of that class are set to the "not ambiguous class" (1)
            predicted_labels[~ambiguous_pronoun_mask] = 1

            # However, it may happen again that we have multiple pronouns classified as ambiguous, 
            # since there may be more than one logit with value = highest_ambiguous_pronoun_logit
        
        
        label_ambiguous_mask = labels == 2
        eval_correct += int(labels[label_ambiguous_mask] == predicted_labels[label_ambiguous_mask])
        
        return eval_correct
    

    def _print_step_log(self, step, running_loss, running_acc):
        print(f'\t| step {step+1:3d}/{len(self.train_dataloader):d} | train_loss: {running_loss:.3f} | ' \
                f'train_acc: {running_acc:.3f} |')

    def _print_epoch_log(self, epoch, epochs, train_loss, valid_loss, valid_acc):
        print('-' * 76)
        print(f'| epoch {epoch+1:>3d}/{epochs:<3d} | train_loss: {train_loss:.3f} | ' \
                f'valid_loss: {valid_loss:.3f} | valid_acc: {valid_acc:.3f} |')
        print('-' * 76)
        
    
    def _save_model(self, task_type, epoch, valid_acc, scaler, metrics_history):
        print("Saving model...")
        params_to_save = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics_history": metrics_history,
        }
        
        if self.scheduler is not None:
            params_to_save["scheduler_state_dict"] = self.scheduler.state_dict()
            
        if scaler is not None:
            params_to_save["scaler_state_dict"] = scaler.state_dict(),
        
        save_path = f"{self.args.output_dir}my_model{str(task_type)}_{str(valid_acc)[2:5]}_{epoch+1}"
        now = datetime.now()
        current_time = now.strftime("%H-%M-%S")
        
        if os.path.exists(f"{save_path}_{current_time}.pth"):
            torch.save(params_to_save, f"{save_path}_{current_time}_new.pth")
        else:
            torch.save(params_to_save, f"{save_path}_{current_time}.pth")

        print("Model saved.")

    def _resume_model(self, path, scaler):
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])


model = GAPModel(model_name_or_path).to(device, non_blocking=True)

# last_frozen_layer = 6

# modules = [model.bert.embeddings, *model.bert.encoder.layer[:last_frozen_layer]]
# for module in modules:
#     for param in module.parameters():
#         param.requires_grad = False

yaml_file = "./train.yaml"
# Read configuration file with all the necessary parameters
with open(yaml_file) as file:
    config = yaml.safe_load(file)
    
training_args = CustomTrainingArguments(**config['training_args'])

# Make sure that the learning rate is read as a number and not as a string
training_args.learning_rate = float(training_args.learning_rate)
print(training_args)

criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device=device, non_blocking=True)
optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
# scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.2)
scheduler = None

batch_size = 4

collator = Collator(device)
train_dataloader = DataLoader(train_ds, batch_size=batch_size, 
                              collate_fn=collator, shuffle=True)
valid_dataloader = DataLoader(valid_ds, batch_size=batch_size, 
                              collate_fn=collator, shuffle=False)

trainer = Trainer(model, training_args, 
                  train_dataloader, valid_dataloader, 
                  criterion, optimizer, scheduler)


trainer.train()



y_true_list = []
y_pred_list = []
logits = []


eval_correct, total_count = 0.0, 0.0
model.eval()
with torch.no_grad():
    collator = Collator(device)
    dataloader = DataLoader(valid_ds, batch_size=1, collate_fn=collator, shuffle=False)
    for features, offsets, labels in dataloader:
        predictions = model(features, offsets)


        predictions = predictions.view(-1, predictions.shape[-1])
        labels = labels.view(-1)

        logits.append(predictions)

        mask = labels != 0
        labels = labels[mask]
        y_true_list.append(labels.tolist())
        
        
        predictions = predictions[mask]
        maximum_logits, predicted_labels = predictions.max(1)

        
        # It may happen that more than one pronoun is classify as ambiguous
        multiple_ambiguous_pronouns_mask = predicted_labels == 2
        ambiguous_pronouns_logits = maximum_logits[multiple_ambiguous_pronouns_mask]
        
        # More than one pronoun is classify as ambiguous
        if len(ambiguous_pronouns_logits) > 1:
            # Get the highest logit among the ambiguous ones
            highest_ambiguous_pronoun_logit = ambiguous_pronouns_logits.max()

            # Identity the position of the logit that should correspond to the ambiguous prononun class (2)
            ambiguous_pronoun_mask = maximum_logits == highest_ambiguous_pronoun_logit

            # All the predictions that are not of that class are set to the "not ambiguous class" (1)
            predicted_labels[~ambiguous_pronoun_mask] = 1

            # However, it may happen again that we have multiple pronouns classified as ambiguous, 
            # since there may be more than one logit with value = highest_ambiguous_pronoun_logit
        
        
        # When the model predicts that all the pronouns are not ambiguous (no class 2)
        if len(predicted_labels) != 0 and not torch.any(predicted_labels == 2):
            # print(predictions)
            # print(predictions[:, 2])
            # Try to select the most probable ambiguous pronoun
            probable_ambiguous_index = predictions[:,-2].argmax()
            predicted_labels[probable_ambiguous_index] = 2
        
        
        y_pred_list.append(predicted_labels.tolist())
        
        
        label_ambiguous_mask = labels == 2
        eval_correct += (labels[label_ambiguous_mask] == predicted_labels[label_ambiguous_mask]).sum().item()
        total_count += 1


print("Real Accuracy: ", eval_correct / total_count)
print("\n")

count = 0
count_wrong = 0
for sentence_id, (true, pred) in enumerate(zip(y_true_list, y_pred_list)):
    for idx, elem in enumerate(true):
        if (elem == 2 and pred[idx] == 1) or (elem == 1 and pred[idx] == 2):
            count_wrong += 1

    count += 1

print("Strong Accuracy: ", 1 - count_wrong / count)