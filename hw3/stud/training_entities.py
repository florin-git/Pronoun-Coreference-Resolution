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


class GAP_AmbiguousDetection_Dataset(Dataset):
    """Custom GAP Dataset class"""

    def __init__(self, df, tokenizer, stanza_processor, tag_labels, labeled=True):
        self.df = df

        self.labeled = labeled
        self.tokenizer = tokenizer
        self.stanza_processor = stanza_processor
        self.tag_labels = tag_labels
        
        self.tokens = []
        self.entities_offsets = []
        self.coreferent_ent_offsets = []
        self.ambiguous_pron_offsets = []
        
        self._convert_tokens_to_ids()

        if labeled:
            self.labels = []
            self._assign_class_to_tokens()
        
            assert len(self.tokens) == len(self.labels)

    
    
        
    def _assign_class_to_tokens(self):
        
        for sentence_idx in range(len(self.entities_offsets)):
            # Merge entites offsets with coreferent_offsets
            all_entities_offsets = sorted(self.entities_offsets[sentence_idx] + 
                                          self.coreferent_ent_offsets[sentence_idx])
        
            
            
            labels = []
            for offsets in all_entities_offsets:
                if offsets == self.coreferent_ent_offsets[sentence_idx]:
                    labels.append(2)
                else:
                    labels.append(1)
                    
            self.labels.append(labels)   
        
    def _convert_tokens_to_ids(self):
        CLS = [self.tokenizer.cls_token]
        SEP = [self.tokenizer.sep_token]
        
        tag_labels = self.tag_labels
        pronoun_tag = tag_labels['pronoun_tag']
        start_entity_tag = tag_labels['start_entity_tag']
        end_entity_tag = tag_labels['end_entity_tag']
        start_coref_entity_tag = tag_labels['start_coref_entity_tag']
        end_coref_entity_tag = tag_labels['end_coref_entity_tag']
        

        for _, row in self.df.iterrows():
            tokens, entities_offsets = self._tokenize(row)
            
            for (start_off, end_off) in zip(entities_offsets[start_entity_tag], entities_offsets[end_entity_tag]):
                
                ambiguos_pronoun = tokens[entities_offsets[pronoun_tag][0]]
                context = tokens + [ambiguos_pronoun, "is"]
                tokens_to_convert = CLS + context + SEP + tokens[start_off:end_off] + SEP
                self.tokens.append(self.tokenizer.convert_tokens_to_ids(tokens_to_convert))

#             tokens_to_convert = CLS + context + SEP + ["neither"] + SEP
#             self.tokens.append(self.tokenizer.convert_tokens_to_ids(tokens_to_convert))

            
            # Because of the introduction of CLS we have to add 1 to the offsets
        
            if pronoun_tag in entities_offsets:
                self.ambiguous_pron_offsets.append(entities_offsets[pronoun_tag][0]+1)
            
            if start_coref_entity_tag in entities_offsets:
                self.coreferent_ent_offsets.append((entities_offsets[start_coref_entity_tag][0]+1,
                                                    entities_offsets[end_coref_entity_tag][0]+1))
            
            self.entities_offsets.append(([off+1 for off in entities_offsets[start_entity_tag]],
                                          [off+1 for off in entities_offsets[end_entity_tag]]))
    
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
        coreferent_entity_offset = self.get_coreferent_entity_offset(row)
        
        # Parse the text through 'stanza'
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

#                 offsets.append((start_off, end_off))
                
                # In order to identify the coreferent entity, I use special tags
                if coreferent_entity_offset != -1 and start_off == current_coreferent_entity_offset:
                    text = self._insert_tag(text, (start_off, end_off), start_coref_entity_tag, end_coref_entity_tag)

                else:
                    text = self._insert_tag(text, (start_off, end_off), start_entity_tag, end_entity_tag)
                
                count_entities += 1
        
#         self.ambiguous_pron_offsets.append(current_pronoun_offset)
#         self.entities_offsets.append(offsets)
        
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
        entities_offsets = {}
        final_tokens = []
        
        tag_labels = self.tag_labels
        start_entity_tag = tag_labels['start_entity_tag']
        end_entity_tag = tag_labels['end_entity_tag']
        start_coref_entity_tag = tag_labels['start_coref_entity_tag']
        end_coref_entity_tag = tag_labels['end_coref_entity_tag']


        text = self._delimit_entities(row)
        
        for token in self.tokenizer.tokenize(text):
            # Replace the special tags with the general entity tags
            if token == start_coref_entity_tag:
                final_tokens.append(start_entity_tag)

            elif token == end_coref_entity_tag:
                final_tokens.append(end_entity_tag)

            else:
                final_tokens.append(token)
            
            if token in [*self.tag_labels.values()]:
                # If end tag, append the index of previous token
                if "/" in token:
                    entities_offsets.setdefault(token, []).append(len(final_tokens) - 1)

                else:
                    entities_offsets.setdefault(token, []).append(len(final_tokens))
        
        
        return final_tokens, entities_offsets

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        if self.labeled:
#             assert len(self.pronouns_offsets[idx]) == len(self.labels[idx])
            return self.tokens[idx], self.entities_offsets[idx], self.coreferent_ent_offsets[idx], self.ambiguous_pron_offsets[idx], self.labels[idx]
        return self.tokens[idx], self.entities_offsets[idx], self.coreferent_ent_offsets[idx], self.ambiguous_pron_offsets[idx], None