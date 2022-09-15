from transformers import (
    AutoTokenizer,
    BertModel,
    logging
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import numpy as np
import pandas as pd
import stanza

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler

import os
import random
import time
import math
import yaml
import dill as pickle
from typing import *
from datetime import datetime
from collections import namedtuple

# from trainer import TokenClassificationTrainer
from arguments import *
# from gap_dataset import *
from gap_utils import *

SEED = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

logging.set_verbosity_error()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yaml_file = "./train.yaml"
# Read configuration file with all the necessary parameters
with open(yaml_file) as file:
    config = yaml.safe_load(file)
    
training_args = CustomTrainingArguments(**config['training_args'])
model_args = ModelArguments(**config['model_args'])
model_name_or_path = model_args.model_name_or_path

class GAP_Entity_Detection_Dataset(Dataset):
    """
    Custom GAP dataset for enitities identification and resolution.
    
    Parameters
    ----------
    df: pd.DataFrame
        A dataframe from the GAP dataset.
        
    tokenizer: PreTrainedTokenizerBase
        The tokenizer used to preprocess the data.
        
    stanza_processor: stanza.Pipeline
        Stanza processor used for NER.
    
    tag_labels: Dict[str, str]
        A dictionary containing as values the tags that will be inserted
        to delimit an entity or a pronoun.
          
    keep_tags: bool
        If true the tags added to text are kept even after
        the tokenization process.
  
    labeled: bool
        If the dataset also contains the labels.
        
    cleaned: bool
        Whether the GAP dataframe is already cleaned or not.
    """

    def __init__(
        self, 
        df: pd.DataFrame, 
        tokenizer: PreTrainedTokenizerBase, 
        stanza_processor: stanza.Pipeline,
        tag_labels: Dict[str, str],
        keep_tags: bool=False, 
        labeled: bool=True, 
        cleaned: bool=True
    ):
        
        if not cleaned:
             self.clean_dataframe(df)
                
        self.df = df
        self.tokenizer = tokenizer
        self.stanza_processor = stanza_processor
        self.keep_tags = keep_tags
        self.labeled = labeled
        
        self.tag_labels = tag_labels
        self._init_tag_labels()
        
        self.samples = []
        self._convert_tokens_to_ids()
        
        
    @staticmethod
    def clean_text(text: str):
        text = text.translate(str.maketrans("`", "'"))
        return text

    def clean_dataframe(self, df: pd.DataFrame):
        df['text'] = df['text'].map(self.clean_text)
        df['entity_A'] = df['entity_A'].map(self.clean_text) 
        df['entity_B'] = df['entity_B'].map(self.clean_text)
        
    def _init_tag_labels(self):
        self.pronoun_tag = self.tag_labels['pronoun_tag']
        self.start_ent_tag = self.tag_labels['start_ent_tag']
        self.end_ent_tag = self.tag_labels['end_ent_tag']
        self.start_coref_ent_tag = self.tag_labels['start_coref_ent_tag']
        self.end_coref_ent_tag = self.tag_labels['end_coref_ent_tag']
        
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

        Sample = namedtuple("Sample", ['tokens', 'start_ents_offsets', 'end_ents_offsets', 
                                       'coreferent_ent_offset', 'ambiguous_pron_gender',  'ambiguous_pron_offset'])
        if self.labeled:
            Sample = namedtuple("Sample", Sample._fields + ("labels",))

        for _, row in self.df.iterrows():
            coreferent_ent_offset_tuple = None
            tokens, entities_offsets = self.tokenize(row)
            
            tokens_to_convert = CLS + tokens + SEP
            sample = {'tokens': self.tokenizer.convert_tokens_to_ids(tokens_to_convert)}
            
            # Because of the introduction of CLS we have to add 1 to the offsets
            if entities_offsets[self.pronoun_tag]:
                pronoun_offset = entities_offsets[self.pronoun_tag][0]
                sample['ambiguous_pron_gender'] = self.get_gender(tokens[pronoun_offset])
                sample['ambiguous_pron_offset'] = pronoun_offset + 1
            
            if self.labeled:
                if entities_offsets[self.start_coref_ent_tag]:
                    coreferent_ent_offset_tuple = (entities_offsets[self.start_coref_ent_tag][0] + 1, 
                                                   entities_offsets[self.end_coref_ent_tag][0] + 1)

                sample['coreferent_ent_offset'] = coreferent_ent_offset_tuple

            start_entities_offsets = [off + 1 for off in entities_offsets[self.start_ent_tag]]
            end_entities_offsets = [off + 1 for off in entities_offsets[self.end_ent_tag]]
            
            if coreferent_ent_offset_tuple is not None:
                # Add coreferent mention offsets
                start_entities_offsets = sorted(start_entities_offsets + [coreferent_ent_offset_tuple[0]])
                end_entities_offsets = sorted(end_entities_offsets + [coreferent_ent_offset_tuple[1]])
                
            if len(start_entities_offsets) == 0:
                raise AssertionError("No entity recognized by stanza!")

            sample['start_ents_offsets'] = start_entities_offsets
            sample['end_ents_offsets'] = end_entities_offsets
    
            if self.labeled:
                sample['labels'] = self._assign_class_to_tokens(start_entities_offsets, coreferent_ent_offset_tuple)

            sample_namedtuple = Sample(**sample)
            self.samples.append(sample_namedtuple)
    
    @staticmethod
    def get_coreferent_entity_and_offset(row: Union[dict, pd.Series]) -> Tuple[str, int]:
        """
        Returns
        -------
            The enitity coreferenced by the pronoun and its offset;
            or (None, -1) if the pronoun does not refer to any of the 
            two mentions.

        Parameters
        ----------
        row: Union[dict, pd.Series]
            A dictionary or a pandas Series containing
            information about enitities offset positions
            and whether they are coreferenced by the pronoun. 
        """

        not_coref_A = row["is_coref_A"] in ["FALSE", "False", False]
        not_coref_B = row["is_coref_B"] in ["FALSE", "False", False]
        if not_coref_A and not_coref_B:
            return None, -1
        is_coref_A = row["is_coref_A"] in ["TRUE", "True", True]
        if is_coref_A:
            return row['entity_A'], row["offset_A"] 
        else:
            return row['entity_B'], row["offset_B"]
    
    @staticmethod
    def get_gender(pronoun: str):
        FEMININE = 0
        MASCULINE = 1
        UNKNOWN = 2
        gender_mapping = {
            'she': FEMININE,
            'her': FEMININE,
            'hers': FEMININE,
            'he': MASCULINE,
            'his': MASCULINE,
            'him': MASCULINE,
        }

        return gender_mapping.get(pronoun.lower(), UNKNOWN)
    
    def get_overwrite_coref_ent_condition(self, start_off, end_off, start_coref_off, coreferent_ent):
        if coreferent_ent is None:
            return False

        end_coref_off = start_coref_off + len(coreferent_ent)

        return (
                   (start_off == start_coref_off or \
                    end_off == end_coref_off) or \

                   # <e>...<c>...</e>...</c>  | <e>...<c>...</c>...</e>
                   (start_off <= start_coref_off and \
                    end_off >= start_coref_off) or \
                   # <c>...<e>...</e>...</c> | <c>...<e>...</e>...</c>
                   (start_coref_off <= start_off and \
                    end_coref_off >= start_off)
               )
    
    def _delimit_entities(self, row):
        
        text = row['text']
        pronoun_offset = row['p_offset']
        coreferent_ent, coreferent_ent_offset = None, -1
        
        # Parse the text using 'stanza'
        doc_processed = self.stanza_processor(text)
        
        # Insert pronoun tag
        text = self._insert_tag(text, (pronoun_offset, None), self.pronoun_tag)
        
        if self.labeled:
            coreferent_ent, coreferent_ent_offset = self.get_coreferent_entity_and_offset(row)
            
            if coreferent_ent is not None:
                if coreferent_ent_offset > pronoun_offset:
                    coreferent_ent_offset += len(self.pronoun_tag)

                # In order to identify the coreferent entity, I use special tags
                text = self._insert_tag(text, (coreferent_ent_offset, coreferent_ent_offset+len(coreferent_ent)), 
                                  self.start_coref_ent_tag, self.end_coref_ent_tag)
    
        entity_already_considered = False
        count_entities = 0
        # Number of characters inserted to delimit an entity
        len_tags = len(self.start_ent_tag) + len(self.end_ent_tag)
  
        
        ner_type = "PERSON"
        people: list = [ent.text for ent in doc_processed.ents if ent.type=="PERSON"]
        # It may happen that stanza does not recognize any PERSON entity in 
        # the text, even if the text always contains at least one PERSON entity.
        # In this case, the analysis will not work; therefore from some
        # experiments, I have noticed that the real PERSON entity is actually
        # recongnize by stanza as an organization ORG.
        if len(people) == 0:
            ner_type = "ORG"
            
            organizations = [ent.text for ent in doc_processed.ents if ent.type=="ORG"]
            # In case there are no organizations either, I simply chose one entity
            # from stanza. 
            # This may happend when the ambiguous pronoun does not
            # refer to any entity in the text and also there are no 
            # PERSON and ORG entities
            if len(organizations) == 0:
                ent_details = [ent for ent in doc_processed.ents][0]
                start_off, end_off = ent_details.start_char, ent_details.end_char
                if start_off > pronoun_offset:
                    start_off += len(self.pronoun_tag)
                    end_off += len(self.pronoun_tag)
                text = self._insert_tag(text, (start_off, end_off), self.start_ent_tag, self.end_ent_tag)
                return text
        
        for ent in doc_processed.ents:
            if ent.type == ner_type:
                # For every tag inserted we have to shift the offsets by the tag length
                start_off = ent.start_char + len_tags*count_entities
                end_off = ent.end_char + len_tags*count_entities
           
                # Because of the new tags, also the pronoun and the coreferent entity offsets are shifted
                current_coreferent_ent_offset = coreferent_ent_offset + len_tags*count_entities
                current_pronoun_offset = pronoun_offset + len_tags*count_entities
                if start_off > current_pronoun_offset:
                    start_off += len(self.pronoun_tag)
                    end_off += len(self.pronoun_tag)

                count_entities += 1
                overwrite_coref_ent_tag = self.get_overwrite_coref_ent_condition(start_off, end_off, 
                                                                            current_coreferent_ent_offset, 
                                                                            coreferent_ent)
        
                if overwrite_coref_ent_tag:
                    entity_already_considered = True
                    continue

                is_entity_not_recognized_by_stanza = coreferent_ent is not None \
                                                     and start_off > current_coreferent_ent_offset \
                                                     and not entity_already_considered
                
                if is_entity_not_recognized_by_stanza:
                    start_off += len(self.start_coref_ent_tag) + len(self.end_coref_ent_tag)
                    end_off += len(self.start_coref_ent_tag) + len(self.end_coref_ent_tag)

                text = self._insert_tag(text, (start_off, end_off), 
                                            self.start_ent_tag, self.end_ent_tag)
        
        return text
        

    def _insert_tag(self, text, offsets, start_tag: str, end_tag: str = None):
        start_off, end_off = offsets 

        # Starting tag only
        if end_tag is None:
            text = text[:start_off] + start_tag + text[start_off:]
            return text

        text = text[:start_off] + start_tag + text[start_off:end_off] + end_tag + text[end_off:]
        return text
    
    
    def tokenize(self, row):
        tokens = []
        tag_labels = self.tag_labels
        entities_offsets = {tag: [] for tag in tag_labels.values()}
        
        text = self._delimit_entities(row)
        
        for token in self.tokenizer.tokenize(text):       
            if token in [*tag_labels.values()]:
                entities_offsets[token].append(len(tokens)) 
                continue
            
            # Replace the special tags with the general entity tags 
            if token == self.start_coref_ent_tag:
                tokens.append(self.start_ent_tag)

            elif token == self.end_coref_ent_tag:
                tokens.append(self.end_ent_tag)

            else:
                tokens.append(token)

        return tokens, entities_offsets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


tag_labels = {
    "pronoun_tag": "<p>",
    "start_ent_tag": "<e>",
    "end_ent_tag": "</e>",
    "start_coref_ent_tag": "<c>",
    "end_coref_ent_tag": "</c>"
}


tokenizer_name_or_path = model_args.tokenizer
if tokenizer_name_or_path is None:
    tokenizer_name_or_path = model_name_or_path

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, never_split=list(tag_labels.values()))
tokenizer.add_tokens(list(tag_labels.values()), special_tokens=True)

train_ds_save_path = "../../model/datasets/bert-base-uncased/train_no_tags.ds"
valid_ds_save_path = "../../model/datasets/bert-base-uncased/valid_no_tags.ds"

with open(train_ds_save_path, 'rb') as file:
    train_ds = pickle.load(file)

with open(valid_ds_save_path, 'rb') as file:
    valid_ds = pickle.load(file)

def compute_max_len(sequences: Union[List[List[int]], Tuple[List[int]]],
                    truncate_len: int) -> int:
    """
    Computes the maximum length in the sequences.         
    """
    max_len = min(
        max((len(x) for x in sequences)),
        truncate_len
    )
    return max_len


def pad_sequence(sequences: Union[List[List[int]], Tuple[List[int]]],
                 max_len: int, pad: int) -> np.ndarray:
    """
    Returns
    -------
        A numpy array padded with the 'pad' value until 
        the 'max_len' length. 

    Parameters
    ----------
    sequences: Union[List[List[int]], Tuple[List[int]]]
        A list or tuple of lists. 

    max_len: int
        The length to which the input is padded.

    pad: int
        The padding value.
    """
    array_sequences = np.full((len(sequences), max_len), pad, dtype=np.int64)

    # Padding
    for i, sequence in enumerate(sequences):
        array_sequences[i, :len(sequence)] = sequence

    return array_sequences

class Collator_Token_Classification:
    """
    Collator for Token Classification.
    
    Returns
    -------
        A dictionary of tensors of the batch sequences in input.

    Parameters
    ----------
    device: str
        Where (CPU/GPU) to load the features.
        
    pad: int
        The padding token.

    truncate_len: int
        Maximum length possible in the batch.

    labeled: bool
        If the batch also contains the labels.
    """
    def __init__(self, device: str, pad: int=0, 
                 truncate_len: int=512, labeled=True):
        self.device = device
        self.pad = pad
        self.truncate_len = truncate_len
        self.labeled = labeled
        
    def __call__(self, batch):
        if self.labeled:
            batch_features, batch_start_ents_offsets, batch_end_ents_offsets, _, ambiguous_pron_gender, ambiguous_pron_offset, batch_labels = zip(*batch)
    
        else:
            batch_features, batch_start_ents_offsets, batch_end_ents_offsets, _, ambiguous_pron_gender, ambiguous_pron_offset = zip(*batch)
    
    
        collate_sample = {}

        max_len_features_in_batch = compute_max_len(batch_features, self.truncate_len)
        max_len_offsets_in_batch = compute_max_len(batch_start_ents_offsets, self.truncate_len)

        # Features        
        padded_features = pad_sequence(batch_features, max_len_features_in_batch, self.pad)
        features_tensor = torch.tensor(padded_features, device=self.device)
        collate_sample['features'] = features_tensor

        # Offsets
        padded_start_ents_offsets = pad_sequence(batch_start_ents_offsets, max_len_offsets_in_batch, self.pad)
        start_ents_offsets_tensor = torch.tensor(padded_start_ents_offsets, device=self.device)
        
        padded_end_ents_offsets = pad_sequence(batch_end_ents_offsets, max_len_offsets_in_batch, self.pad)
        end_ents_offsets_tensor = torch.tensor(padded_end_ents_offsets, device=self.device)

        collate_sample['entities_offsets'] = list(zip(start_ents_offsets_tensor, end_ents_offsets_tensor))
        collate_sample['pronouns_offset'] = torch.tensor(ambiguous_pron_offset, dtype=torch.int64, device=self.device)
       
        if not self.labeled:
            return collate_sample
        
        # Labels
        padded_labels = pad_sequence(batch_labels, max_len_offsets_in_batch, self.pad)
        labels_tensor = torch.tensor(padded_labels, dtype=torch.uint8, device=self.device)
        collate_sample['labels'] = labels_tensor
        
        return collate_sample


class Entity_Resolution_Head(nn.Module):
    def __init__(self, bert_hidden_size: int, args: ModelArguments):
        super().__init__()
        
        self.args = args
        self.bert_hidden_size = bert_hidden_size

        input_size_pronoun = bert_hidden_size
        input_size_entities = bert_hidden_size * 3
        if args.output_strategy == "concat":
            input_size_pronoun *= 4
            input_size_entities *=4
        
        self.ffnn_pronoun = nn.Sequential(
            nn.Linear(input_size_pronoun, bert_hidden_size),
            nn.LayerNorm(bert_hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(bert_hidden_size, args.head_hidden_size),
        )
        
        self.ffnn_entities = nn.Sequential(
            nn.Linear(input_size_entities, bert_hidden_size),
            nn.LayerNorm(bert_hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(bert_hidden_size, args.head_hidden_size),
        )
        
        bilinear_hidden_size = 512
        
#         self.gender_embedding = torch.nn.Embedding(2, 10)
#         self.ffnn_pronoun = nn.Linear(bert_hidden_size, linear_hidden_size)
#         self.ffnn_entities = nn.Linear(bert_hidden_size * 2 +10, linear_hidden_size)

        
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.LeakyReLU()
        
        # self.bilinear = torch.nn.Bilinear(args.head_hidden_size, args.head_hidden_size, bilinear_hidden_size, bias=False)
        self.linear1 = torch.nn.Linear(args.head_hidden_size*2, bilinear_hidden_size)
        # self.linear = torch.nn.Linear(bilinear_hidden_size, bilinear_hidden_size)
        
        self.classifier = nn.Linear(bilinear_hidden_size, args.num_output)
        
        # self._init_linear_weights(0.5)

    def forward(self, bert_outputs, entities_offsets, pronouns_offset):
        pronouns_embeddings, entities_embeddings = self._retrieve_embeddings(bert_outputs, entities_offsets, pronouns_offset)
        
    
        x_ent = self.ffnn_entities(entities_embeddings)
        x_pron = self.ffnn_pronoun(pronouns_embeddings)
        x_pron = x_pron.unsqueeze_(dim=1).expand(-1, x_ent.shape[1], -1)
        
        x = self.linear1(torch.cat([x_pron, x_ent], dim=-1))
        # x = self.bilinear(x_pron, x_ent) + self.linear(x_ent)
        x = self.relu(x)
        x = self.dropout(x)

        output = self.classifier(x)
        return output
    
    def _init_linear_weights(self, initrange):
        for module in [self.ffnn_entities, self.ffnn_pronoun, self.linear1]:
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight,initrange, initrange)
                nn.init.constant_(module.bias, 0)
        
        self.classifier.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()
    

    def _retrieve_embeddings(self, bert_embeddings, entities_offsets, pronouns_offset):
        all_embeddings = []
        pronouns_embeddings = []
        
        # Consider embeddings and offsets in each batch separately
        for embeddings, ent_offs, pron_off in zip(bert_embeddings, entities_offsets, pronouns_offset):
            batch_embeddings = []

            pronoun_embedding = embeddings[pron_off]
            pronouns_embeddings.append(pronoun_embedding)
            for start, end in zip(*ent_offs):
                if (start, end) == (0, 0): # Dealing with padding
                    batch_embeddings.append(torch.zeros(embeddings.shape[-1] * 3, device=device))
                    continue

                start_entity_embedding = embeddings[start]
                end_entity_embedding = embeddings[end-1]
                mean_entity_embedding = embeddings[start:end].mean(dim=0)
                
                entity_embedding = torch.cat([start_entity_embedding, end_entity_embedding, mean_entity_embedding], dim=-1)
                batch_embeddings.append(entity_embedding)

            all_embeddings.append(torch.stack(batch_embeddings, dim=0))

        # shape: batch_size x seq_length x (embed_dim * 3)
        merged_embeddings = torch.stack(all_embeddings, dim=0)
        # shape: batch_size x embed_dim
        stacked_pronouns_embeddings = torch.stack(pronouns_embeddings, dim=0)
        return stacked_pronouns_embeddings, merged_embeddings

class CR_Model(nn.Module):
    """The main model."""

    def __init__(self, bert_model: str, tokenizer, args: ModelArguments):
        super().__init__()

        self.args = args
        
        if bert_model in {"bert-base-uncased", "bert-base-cased"}:
            self.bert_hidden_size = 768
        elif bert_model in {"bert-large-uncased", "bert-large-cased"}:
            self.bert_hidden_size = 1024
        else:
            raise ValueError("Unsupported BERT model.")

        self.bert = BertModel.from_pretrained(
            bert_model).to(device, non_blocking=True)

        # If the tag tokens (e.g., <p>, <a> etc.) are present in the features,
        # the embedding dimension of the bert embeddings must be changed
        # to be compliant with the new size of the tokenizer vocabulary. 
        if args.resize_embeddings:
            self.bert.resize_token_embeddings(len(tokenizer.vocab))
            
        self.head = Entity_Resolution_Head(self.bert_hidden_size, self.args).to(
            device, non_blocking=True)

    def forward(self, sample):
        x = sample['features']
        x_offsets = sample['entities_offsets']
#         x_pronouns_gender = sample['pronouns_gender']
        x_pronouns_offset = sample['pronouns_offset']

        bert_outputs = self.bert(
            x, attention_mask=(x > 0).long(),
            token_type_ids=None, output_hidden_states=True)

        if self.args.output_strategy == "last":
            out = bert_outputs.last_hidden_state

        elif self.args.output_strategy == "concat":
            out = torch.cat([bert_outputs.hidden_states[x] for x in [-1, -2, -3, -4]], dim=-1)

        elif self.args.output_strategy == "sum":
            layers_to_sum = torch.stack([bert_outputs.hidden_states[x] for x in [-1, -2, -3, -4]], dim=0)
            out = torch.sum(layers_to_sum, dim=0)

        else:
            raise ValueError("Unsupported output strategy.")
        
        head_outputs = self.head(out, x_offsets, x_pronouns_offset)
        return head_outputs


class TokenClassificationTrainer:    
    def __init__(
        self,
        device: str,
        model: nn.Module,
        args: CustomTrainingArguments,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        criterion: torch.nn,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        pad: int = 0,
    ):
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.pad = pad
        
        assert args is not None, "No training arguments passed!"
        self.args = args
        
    def train(self):
        args = self.args
        valid_dataloader = self.valid_dataloader
        epochs = args.num_train_epochs
        
        train_losses = []
        train_acc_list = []
        valid_losses = []
        valid_acc_list = []
        
        if args.use_early_stopping:
            patience_counter = 0 

        scaler = GradScaler() if args.use_scaler else None

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
                self._print_sceduler_lr()
                self.scheduler.step()

            self._print_epoch_log(epoch, epochs, train_loss, valid_loss, valid_acc)

            if args.use_early_stopping and len(valid_acc_list) >= 2:
                stop, patience_counter = self._early_stopping(patience_counter, epoch, valid_acc_list)
                if stop:
                    break
        
        training_time = time.time() - training_start_time
        print(f'Training time: {self._print_time(training_time)}')

        metrics_history = {
            "train_losses": train_losses,
            "train_acc": train_acc_list,
            "valid_losses": valid_losses,
            "valid_acc": valid_acc_list,
        }

#         print(metrics_history)
        if args.save_model:
            self._save_model(args.task_type, epoch, valid_acc, scaler, metrics_history)
    
        return metrics_history

    def _inner_training_loop(self, scaler):
        args = self.args
        train_dataloader = self.train_dataloader
        
        train_loss = 0.0
        train_correct, total_count = 0.0, 0.0

        self.model.train()
        for step, sample in enumerate(train_dataloader):
            ### Empty gradients ###
            self.optimizer.zero_grad(set_to_none=True)

            ### Forward ###
            if scaler is None:
                predictions = self.model(sample)
                labels = sample['labels']
                train_correct, total_count = self.compute_metrics(predictions, labels, 
                                                              train_correct, total_count)
                labels = labels.view(-1)
                predictions = predictions.view(-1, predictions.shape[-1])
                loss = self.criterion(predictions, labels)
                loss = torch.nanmean(loss)
            else:
                 
                with torch.autocast(device_type=self.device):
                    predictions = self.model(sample)
                    labels = sample['labels']
                    train_correct, total_count = self.compute_metrics(predictions, labels, 
                                                            train_correct, total_count)
                    
                    labels = labels.view(-1)
                    predictions = predictions.view(-1, predictions.shape[-1])
                    loss = self.criterion(predictions, labels)
                    # if torch.isnan(torch.mean(loss)):
                    #     print(sample, predictions, labels, loss) 
                    # Since there are some nan in the output model,
                    # I decided to compute the mean with nanmean,
                    # so without using the mean in the criterion
                    loss = torch.nanmean(loss)

            ### Backward  ###
            if scaler is None:
                loss.backward()
            else: 
                # Backward pass without mixed precision
                # It's not recommended to use mixed precision for backward pass
                # Because we need more precise loss
                scaler.scale(loss).backward()
            
            if args.grad_clipping is not None:
                if scaler is not None:
                    scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clipping)
            
            ### Update weights ### 
            if scaler is None:
                self.optimizer.step()
            else:
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
                labels = sample['labels']
                eval_correct, total_count = self.compute_metrics(predictions, labels, 
                                                                 eval_correct, total_count)
                
                labels = labels.view(-1)
                predictions = predictions.view(-1, predictions.shape[-1])
                loss = self.criterion(predictions, labels)
                loss = torch.nanmean(loss)
                valid_loss += loss.item()

        
        return valid_loss / len(eval_dataloader), eval_correct / total_count

    def compute_metrics(self, predictions, labels, num_correct, total_count):  
        # Iterate one batch at a time
        for one_batch_predictions, one_batch_labels in zip(predictions, labels):
            num_batch_correct, batch_count = 0.0, 0.0

            mask = one_batch_labels != self.pad
            one_batch_labels = one_batch_labels[mask]

            one_batch_predictions = one_batch_predictions[mask]
            maximum_logits, predicted_labels = one_batch_predictions.max(1)

            # It may happen that more than one entity is classify as the coreferent one
            multiple_coreferent_entities_mask = predicted_labels == 2
            coreferent_entities_logits = maximum_logits[multiple_coreferent_entities_mask]

            # More than one pronoun is classify as ambiguous
            if len(coreferent_entities_logits) > 1:
                # Get the highest logit among the coreferent ones
                highest_coreferent_entity_logit = coreferent_entities_logits.max()

                # Identity the position of the logit that should correspond to the coreferent_entity class (2)
                coreferent_entity_mask = maximum_logits == highest_coreferent_entity_logit

                # All the predictions that are not of that class are set to the "generic entity class" (1)
                predicted_labels[~coreferent_entity_mask] = 1

                # However, it may happen again that we have multiple entities classified as coreferent one, 
                # since there may be more than one logit with value = highest_coreferent_entity_logit

            label_coreferent_mask = one_batch_labels == 2
            num_batch_correct += (one_batch_labels[label_coreferent_mask] == predicted_labels[label_coreferent_mask]).sum().item()
            batch_count += 1
        
        num_correct += num_batch_correct
        total_count += batch_count
    
        return num_correct, total_count

    def _early_stopping(self, patience_counter, epoch, valid_acc_list):
        args = self.args

        # stop = args.early_stopping_mode == 'min' and epoch > 0 and valid_acc_list[-1] > valid_acc_list[-2]
        stop = args.early_stopping_mode == 'max' and epoch > 0 and valid_acc_list[-1] < valid_acc_list[-2]
        if stop:
            if patience_counter >= args.early_stopping_patience:
                print('Early stop.')
                return stop, patience_counter
            else:
                print('-- Patience.\n')
                patience_counter += 1

        return False, patience_counter   
    
    def _print_time(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def _print_sceduler_lr(self):
        print('-' * 17)
        print(f"| LR: {self.scheduler.get_last_lr()[0]:.3e} |")

    def _print_step_log(self, step, running_loss, running_acc):
        print(f'\t| step {step+1:4d}/{len(self.train_dataloader):d} | train_loss: {running_loss:.3f} | ' \
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
            params_to_save["scaler_state_dict"] = scaler.state_dict()
            
        save_path = f"{self.args.output_dir}my_model{str(task_type)}_{str(valid_acc)[2:5]}_{epoch+1}"
        now = datetime.now()
        current_time = now.strftime("%H-%M-%S")
        
        if os.path.exists(f"{save_path}_{current_time}.pth"):
            torch.save(params_to_save, f"{save_path}_{current_time}_new.pth")
        else:
            torch.save(params_to_save, f"{save_path}_{current_time}.pth")
        
        print("Model saved.") 

def freeze_weights(modules):
    for module in modules:
        for param in module.parameters():
            if hasattr(param, 'requires_grad'):
                param.requires_grad = False

model = CR_Model(model_name_or_path, tokenizer, model_args).to(device, non_blocking=True)

# last_frozen_layer = 12
# modules = [model.bert.embeddings, *model.bert.encoder.layer[:last_frozen_layer]]
# # modules = [*model.bert.encoder.layer[:last_frozen_layer]]
# freeze_weights(modules)

batch_size = 4

collator = Collator_Token_Classification(device)
train_dataloader = DataLoader(train_ds, batch_size=batch_size, 
                              collate_fn=collator, shuffle=True)
valid_dataloader = DataLoader(valid_ds, batch_size=batch_size, 
                              collate_fn=collator, shuffle=False)

# Make sure that the learning rate is read as a number and not as a string
training_args.learning_rate = float(training_args.learning_rate)

criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0, 0.4, 0.6]), 
                                      ignore_index=0, reduction="none").to(device=device, non_blocking=True)
optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
# scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
scheduler = None


trainer = TokenClassificationTrainer(str(device), model, training_args, 
                  train_dataloader, valid_dataloader, 
                  criterion, optimizer, scheduler)

trainer.train()


y_true_list = []
y_pred_list = []
logits = []


eval_correct, total_count = 0.0, 0.0
model.eval()
with torch.no_grad():
    collator = Collator_Token_Classification(device)
    dataloader = DataLoader(valid_ds, batch_size=1, collate_fn=collator, shuffle=False)
    for idx, sample in enumerate(dataloader):
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

#             However, it may happen again that we have multiple pronouns classified as ambiguous, 
#             since there may be more than one logit with value = highest_ambiguous_pronoun_logit
        
        
        y_pred_list.append(predicted_labels.tolist())
        
        
        label_ambiguous_mask = labels == 2
        eval_correct += (labels[label_ambiguous_mask] == predicted_labels[label_ambiguous_mask]).sum().item()
        total_count += 1


print("Accuracy: ", eval_correct / total_count)