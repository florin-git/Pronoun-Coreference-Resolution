from transformers import (
    Trainer,
    BertTokenizer,
    AutoTokenizer,
    BertConfig,
    BertModel,
)
import transformers
import numpy as np
import pandas as pd
import stanza

import os
import random
import time
import math
import yaml
import dill as pickle
from typing import *
from datetime import datetime
from collections import namedtuple
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


class GAP_AmbiguousDetection_Dataset(Dataset):
    """Custom GAP Dataset class"""

    def __init__(self, df, tokenizer, stanza_processor, tag_labels, labeled=True):
        self.df = df

        self.labeled = labeled
        self.tokenizer = tokenizer
        self.stanza_processor = stanza_processor
        self.tag_labels = tag_labels
        
        self.samples = []
        self._convert_tokens_to_ids()
        
        
    def _assign_class_to_tokens(self, entities_offsets, coreferent_ent_offset):
        labels = []
        for offset in entities_offsets:
            if coreferent_ent_offset is not None and offset == coreferent_ent_offset[0]:
                labels.append(2)
            else:
                labels.append(1)
                
        return labels  
        
    def _convert_tokens_to_ids(self):
        CLS = [self.tokenizer.cls_token]
        SEP = [self.tokenizer.sep_token]
        
        tag_labels = self.tag_labels
        pronoun_tag = tag_labels['pronoun_tag']
        start_entity_tag = tag_labels['start_entity_tag']
        end_entity_tag = tag_labels['end_entity_tag']
        start_coref_entity_tag = tag_labels['start_coref_entity_tag']
        end_coref_entity_tag = tag_labels['end_coref_entity_tag']
        
        Sample = namedtuple("Sample", ['tokens', 'start_entities_offsets', 'end_entities_offsets', 'coreferent_ent_offset', 'ambiguous_pron_offset'])
        if self.labeled:
            Sample = namedtuple("Sample", Sample._fields + ("labels",))

        for _, row in self.df.iterrows():
            coreferent_ent_offset_tuple = None
            tokens, entities_offsets = self._tokenize(row)
            tokens_to_convert = CLS + tokens + SEP
            sample = {'tokens': self.tokenizer.convert_tokens_to_ids(tokens_to_convert)}
            
            # Because of the introduction of CLS we have to add 1 to the offsets
            if entities_offsets[pronoun_tag]:
                sample['ambiguous_pron_offset'] = entities_offsets[pronoun_tag][0] + 1
            if entities_offsets[start_coref_entity_tag]:
                coreferent_ent_offset_tuple = (entities_offsets[start_coref_entity_tag][0] + 1, 
                                               entities_offsets[end_coref_entity_tag][0] + 1)
 
            sample['coreferent_ent_offset'] = coreferent_ent_offset_tuple

            start_entities_offsets = [off + 1 for off in entities_offsets[start_entity_tag]]
            end_entities_offsets = [off + 1 for off in entities_offsets[end_entity_tag]]

            
            # all_entities_offsets = list(zip([off + 1 for off in entities_offsets[start_entity_tag]], 
            #                                 [off + 1 for off in entities_offsets[end_entity_tag]]))

            if coreferent_ent_offset_tuple is not None:
                # Add coreferent mention offsets
                start_entities_offsets = sorted(start_entities_offsets + [coreferent_ent_offset_tuple[0]])
                end_entities_offsets = sorted(end_entities_offsets + [coreferent_ent_offset_tuple[1]])
                # all_entities_offsets = sorted(all_entities_offsets + [coreferent_ent_offset_tuple])
            
            # sample['all_entities_offsets'] = all_entities_offsets
            sample['start_entities_offsets'] = start_entities_offsets
            sample['end_entities_offsets'] = end_entities_offsets
            


            if self.labeled:
                # sample['labels'] = self._assign_class_to_tokens(all_entities_offsets, coreferent_ent_offset_tuple)
                sample['labels'] = self._assign_class_to_tokens(start_entities_offsets, coreferent_ent_offset_tuple)

            sample_namedtuple = Sample(**sample)
            self.samples.append(sample_namedtuple)
    
    @staticmethod
    def get_coreferent_entity_offset(row):
        not_coref_A = row["is_coref_A"] in ["FALSE", False]
        not_coref_B = row["is_coref_B"] in ["FALSE", False]
        if not_coref_A and not_coref_B:
            return -1
        is_coref_A = row["is_coref_A"] in ["TRUE", True]
        return row["offset_A"] if is_coref_A else row["offset_B"]
    
    def _delimit_entities(self, row):
        
        text = row['text']
        pronoun_offset = row['p_offset']

        coreferent_entity_offset = -1
        if self.labeled:
            coreferent_entity_offset = self.get_coreferent_entity_offset(row)
        
        # Parse the text using 'stanza'
        doc_processed = self.stanza_processor(text)
        
        tag_labels = self.tag_labels
        pronoun_tag = tag_labels['pronoun_tag']
        start_entity_tag = tag_labels['start_entity_tag']
        end_entity_tag = tag_labels['end_entity_tag']
        start_coref_entity_tag = tag_labels['start_coref_entity_tag']
        end_coref_entity_tag = tag_labels['end_coref_entity_tag']
         
#         offsets = []
        count_entities = 0

        # Insert pronoun tag
        text = self._insert_tag(text, (pronoun_offset, None), pronoun_tag)

        # Number of characters inserted to delimit an entity
        len_tags = len(start_entity_tag) + len(end_entity_tag)

        for ent in doc_processed.ents:
            if ent.type == "PERSON":
                # For every tag inserted we have to shift the offsets by the tag length
                start_off = ent.start_char + len_tags*count_entities
                end_off = ent.end_char + len_tags*count_entities

                
                # Because of the new tags, also the pronoun and the coreferent entity offsets are shifted
                current_coreferent_entity_offset = coreferent_entity_offset + len_tags*count_entities
                current_pronoun_offset = pronoun_offset + len_tags*count_entities
                if start_off > current_pronoun_offset:
                    start_off += len(pronoun_tag)
                    end_off += len(pronoun_tag)
                    current_coreferent_entity_offset += len(pronoun_tag)

#                 offsets.append((start_off, end_off))
                
                # In order to identify the coreferent entity, I use special tags
                if coreferent_entity_offset != -1 and start_off == current_coreferent_entity_offset:
                    text = self._insert_tag(text, (start_off, end_off), start_coref_entity_tag, end_coref_entity_tag)

                else:
                    text = self._insert_tag(text, (start_off, end_off), start_entity_tag, end_entity_tag)
                
                count_entities += 1
        
        return text
        

    def _insert_tag(self, text, offsets, start_tag: str, end_tag: str = None):
        start_off, end_off = offsets 

        # Starting tag only
        if end_tag is None:
            text = text[:start_off] + start_tag + text[start_off:]
            return text

        text = text[:start_off] + start_tag + text[start_off:end_off] + end_tag + text[end_off:]
        return text

    def _tokenize(self, row):

        final_tokens = []
        tag_labels = self.tag_labels
        entities_offsets = {tag: [] for tag in tag_labels.values()}
        
        start_entity_tag = tag_labels['start_entity_tag']
        end_entity_tag = tag_labels['end_entity_tag']
        start_coref_entity_tag = tag_labels['start_coref_entity_tag']
        end_coref_entity_tag = tag_labels['end_coref_entity_tag']


        text = self._delimit_entities(row)
        
        # for token in self.tokenizer.tokenize(text):
        #     # Replace the special tags with the general entity tags
        #     if token == start_coref_entity_tag:
        #         final_tokens.append(start_entity_tag)

        #     elif token == end_coref_entity_tag:
        #         final_tokens.append(end_entity_tag)

        #     else:
        #         final_tokens.append(token)
            
        #     if token in [*self.tag_labels.values()]:
        #         # If end tag, append the index of previous token
        #         if "/" in token:
        #             entities_offsets[token].append(len(final_tokens) - 1)

        #         else:
        #             entities_offsets[token].append(len(final_tokens))
        for token in self.tokenizer.tokenize(text):       
            if token in [*tag_labels.values()]:
                entities_offsets[token].append(len(final_tokens)) 
                continue
            
            # Replace the special tags with the general entity tags 
            if token == start_coref_entity_tag:
                final_tokens.append(start_entity_tag)

            elif token == end_coref_entity_tag:
                final_tokens.append(end_entity_tag)

            else:
                final_tokens.append(token)


        return final_tokens, entities_offsets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

tag_labels = {
    "pronoun_tag": "<P>",
    "start_entity_tag": "<E>",
    "end_entity_tag": "</E>",
    "start_coref_entity_tag": "<C>",
    "end_coref_entity_tag": "</C>"
}

model_name_or_path = "../../model/tokenizer/"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

train_ds_save_path = "../../model/datasets/train_no_tags.ds"
valid_ds_save_path = "../../model/datasets/valid_no_tags.ds"

with open(train_ds_save_path, 'rb') as file:
    train_ds = pickle.load(file)

with open(valid_ds_save_path, 'rb') as file:
    valid_ds = pickle.load(file)

class Collator:
    def __init__(self, device, labeled=True):
        self.device = device
        self.labeled = labeled
        
    def __call__(self, batch, truncate_len=512):
        """Batch preparation.

        1. Pad the sequences
        2. Transform the target.
        """

        # ['tokens', 'all_entities_offsets', 'coreferent_ent_offset', 'ambiguous_pron_offset', 'labels']
        if self.labeled:
            batch_features, batch_start_entities_offsets, batch_end_entities_offsets, _, _, batch_labels = zip(*batch)
        
        else:
            batch_features, batch_start_entities_offsets, batch_end_entities_offsets, _, _ = zip(*batch)
        
        collate_sample = {}

        max_len_features_in_batch = self.compute_max_len(batch_features, truncate_len)
        max_len_offsets_in_batch = self.compute_max_len(batch_start_entities_offsets, truncate_len)

        # Features        
        padded_features = self.pad_sequence(batch_features, max_len_features_in_batch, 0)
        features_tensor = torch.tensor(padded_features, device=device)
        collate_sample['features'] = features_tensor

        # Offsets
        padded_start_entities_offsets = self.pad_sequence(batch_start_entities_offsets, max_len_offsets_in_batch, 0)
        start_entities_offsets_tensor = torch.tensor(padded_start_entities_offsets, device=device)
        # collate_sample['start_entities_offsets'] = start_entities_offsets_tensor

        padded_end_entities_offsets = self.pad_sequence(batch_end_entities_offsets, max_len_offsets_in_batch, 0)
        end_entities_offsets_tensor = torch.tensor(padded_end_entities_offsets, device=device)
        # collate_sample['end_entities_offsets'] = end_entities_offsets_tensor

        collate_sample['entities_offsets'] = list(zip(start_entities_offsets_tensor, end_entities_offsets_tensor))

        # Labels
        if not self.labeled:
            return collate_sample

        padded_labels = self.pad_sequence(batch_labels, max_len_offsets_in_batch, 0)
        labels_tensor = torch.tensor(padded_labels, dtype=torch.uint8, device=device)
        collate_sample['labels'] = labels_tensor
        
        return collate_sample
    
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
        # self.head_hidden_size = 512
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
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            # nn.Linear(self.head_hidden_size, self.head_hidden_size),
            # nn.LeakyReLU(),
            # nn.Dropout(0.1),
            # nn.Linear(self.head_hidden_size // 2, self.head_hidden_size // 2),
            # nn.LeakyReLU(),
            # nn.Dropout(0.1),
        )

        # lstm_hidden_dim = 256
        # bidirectional = True
        # self.lstm = nn.LSTM(self.head_hidden_size, 
        #                     lstm_hidden_dim, 
        #                     bidirectional=bidirectional,
        #                     num_layers=2,
        #                     dropout=0.1,
        #                     batch_first=True)
        
        # lstm_output_dim = lstm_hidden_dim if bidirectional is False \
        #                     else lstm_hidden_dim * 2

        # self.relu = nn.LeakyReLU()

        
        self.classifier = nn.Linear(self.head_hidden_size, 3)
        # self._init_linear_weights(1.5)

    def forward(self, bert_outputs, offsets):
        embeddings = self._retrieve_entities_embeddings(bert_outputs, offsets)

        x = self.fc(embeddings)
        # x, _ = self.lstm(x)
        # x = self.relu(x)

        output = self.classifier(x)
        return output
        

    def _init_linear_weights(self, init_range):
        for module in self.fc:
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, -init_range, init_range)
                nn.init.constant_(module.bias, 0)
        
        self.classifier.weight.data.uniform_(-init_range, init_range)
        self.classifier.bias.data.zero_()

    def _retrieve_entities_embeddings(self, bert_embeddings, entities_offsets):
        batch_embeddings = []

        # Consider embeddings and offsets in each batch separately
        for embeddings, offsets in zip(bert_embeddings, entities_offsets):
            entities_embeddings = []

            for start, end in zip(*offsets):
                if (start, end) == (0, 0): # Dealing with padding
                    entities_embeddings.append(torch.zeros(embeddings.shape[-1], device=device))
                else:
                    # The embedding of an entity is the mean of all the subtokens embeddings that represent it 
                    entities_embeddings.append(embeddings[start:end].mean(dim=0))

            batch_embeddings.append(torch.stack(entities_embeddings, dim=0))

        # Merge outputs
        merged_entities_embeddings = torch.stack(batch_embeddings, dim=0)

        # shape: batch_size x seq_length x embedding_dim
        return merged_entities_embeddings

class GAPModel(nn.Module):
    """The main model."""

    def __init__(self, bert_model: str, tokenizer):
        super().__init__()

        if bert_model in {"bert-base-uncased", "bert-base-cased"}:
            self.bert_hidden_size = 768
        elif bert_model in {"bert-large-uncased", "bert-large-cased"}:
            self.bert_hidden_size = 1024
        else:
            raise ValueError("Unsupported BERT model.")

        self.bert = BertModel.from_pretrained(bert_model).to(device, non_blocking=True)

        embedding_layer = self.bert.embeddings.word_embeddings

        old_num_tokens, old_embedding_dim = embedding_layer.weight.shape

        # Creating new embedding layer with more entries
        new_embeddings = nn.Embedding(
            len(tokenizer.vocab), old_embedding_dim
        )

        # Setting device and type accordingly
        new_embeddings.to(
            embedding_layer.weight.device,
            dtype=embedding_layer.weight.dtype,
        )

        # Copying the old entries
        new_embeddings.weight.data[:old_num_tokens, :] = embedding_layer.weight.data[
            :old_num_tokens, :
        ]

        self.bert.embeddings.word_embeddings = new_embeddings
        self.head = CorefHead(self.bert_hidden_size).to(device, non_blocking=True)

    def forward(self, sample):
        x = sample['features']
        x_offsets = sample['entities_offsets']

        bert_outputs = self.bert(
            x, attention_mask=(x > 0).long(),
            token_type_ids=None, output_hidden_states=True)

        # out = bert_outputs.last_hidden_state

        layers_to_sum = torch.stack([bert_outputs.hidden_states[x] for x in [-1, -2, -3, -4]], dim=0)
        out = torch.sum(layers_to_sum, dim=0)

        # out = torch.cat([bert_outputs.hidden_states[x] for x in [-1, -2, -3, -4]], dim=-1)

        head_outputs = self.head(out, x_offsets)
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
        print(f'Training time: {self._print_time(training_time)}')

        metrics_history = {
            "train_losses": train_losses,
            "train_acc": train_acc_list,
            "valid_losses": valid_losses,
            "valid_acc": valid_acc_list,
        }
        print(metrics_history)
        if args.save_model:
            self._save_model("2", epoch, valid_acc, scaler, metrics_history)
    
        return #metrics_history

    def _inner_training_loop(self, scaler):
        args = self.args
        train_dataloader = self.train_dataloader
        
        train_loss = 0.0
        train_correct, total_count = 0.0, 0.0

        self.model.train()
        for step, sample in enumerate(train_dataloader):
            # Empty gradients
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward
            with torch.cuda.amp.autocast(): # autocast as a context manager
                predictions = self.model(sample)
                predictions = predictions.view(-1, predictions.shape[-1])
                labels = sample['labels']
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
            for sample in eval_dataloader:
                
                predictions = self.model(sample)
                
                predictions = predictions.view(-1, predictions.shape[-1])
                labels = sample['labels']
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
    
    def _print_time(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

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


model_name_or_path = "bert-base-cased"
# model_name_or_path = "spanbert-base-cased"

model = GAPModel(model_name_or_path, tokenizer).to(device, non_blocking=True)

# first_frozen_layer = -3
# last_frozen_layer = len(model.bert.encoder.layer)

# modules = [model.bert.embeddings, *model.bert.encoder.layer[first_frozen_layer:last_frozen_layer]]
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
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.3)
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
    for sample in dataloader:
        if sample['labels'].numel() == 0:
            continue
        
        predictions = model(sample)
        predictions = predictions.view(-1, predictions.shape[-1])
        labels = sample['labels']
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
        
        
        # # When the model predicts that all the pronouns are not ambiguous (no class 2)
        # if not torch.any(predicted_labels == 2):
        #     # Try to select the most probable ambiguous pronoun
        #     probable_ambiguous_index = predictions[:,-1].argmax()
        #     predicted_labels[probable_ambiguous_index] = 2
        
        
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