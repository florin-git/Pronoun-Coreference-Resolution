from typing import *

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from collections import namedtuple


class GAP_Dataset(Dataset):
    """
    Custom GAP dataset for multiple choice or sequence classfication.

    Parameters
    ----------
    df: pd.DataFrame
        A dataframe from the GAP dataset.

    tokenizer: PreTrainedTokenizerBase
        The tokenizer used to preprocess the data.

    tag_labels: Dict[str, str]
        A dictionary containing as values the tags that will be inserted
        to delimit an entity or a pronoun.

    multiple_choice: bool
        Whether to instantiate a dataset for multiple choice
        classification or not.

    keep_tags: bool
        If true the tags added to text are kept even after
        the tokenization process.

    truncate_up_to_pron: bool
        If false the text is not truncated. 
        Otherwise, the text will be truncated at the last tag inserted, 
        that can delimit either an entity or the amiguous pronoun, 
        with the addition of the ambigous pronoun at the end.

    labeled: bool
        If the dataset also contains the labels.

    cleaned: bool
        Whether the GAP dataframe is already cleaned or not.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        tag_labels: Dict[str, str],
        multiple_choice: bool = False,
        keep_tags: bool = False,
        truncate_up_to_pron: bool = True,
        labeled: bool = True,
        cleaned: bool = True
    ):

        if not cleaned:
            self.clean_dataframe(df)

        self.df = df
        self.tokenizer = tokenizer
        self.multiple_choice = multiple_choice
        self.keep_tags = keep_tags
        self.truncate_up_to_pron = truncate_up_to_pron
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
        self.start_A_tag = self.tag_labels['start_A_tag']
        self.end_A_tag = self.tag_labels['end_A_tag']
        self.start_B_tag = self.tag_labels['start_B_tag']
        self.end_B_tag = self.tag_labels['end_B_tag']

    @staticmethod
    def get_class_id(is_coref_A: Union[str, bool],
                     is_coref_B: Union[str, bool]) -> int:
        """
        Returns
        -------
            An integer representing the class of an input sentence.
            The class id is:
            - 0 if the pronoun references entity A
            - 1 if the pronoun references entity B
            - 2 if the pronoun references neither A nor B
        """
        if is_coref_A in ["TRUE", "True", True]:
            return 0
        elif is_coref_B in ["TRUE", "True", True]:
            return 1
        else:
            return 2

    def _convert_tokens_to_ids(self):
        CLS = [self.tokenizer.cls_token]
        SEP = [self.tokenizer.sep_token]

        Sample = namedtuple("Sample", ['tokens', 'offsets'])
        if self.labeled:
            Sample = namedtuple("Sample", Sample._fields + ("labels",))

        for _, row in self.df.iterrows():
            tokens, offsets = self._tokenize(row)

            if self.multiple_choice:
                final_tokens = self._create_multiple_choices(tokens, offsets)
            else:
                tokens_to_convert = CLS + tokens + SEP
                final_tokens = self.tokenizer.convert_tokens_to_ids(
                    tokens_to_convert)

            sample = {'tokens': final_tokens,
                      'offsets': self._get_offsets_list(offsets)}

            if self.labeled:
                sample['labels'] = self.get_class_id(
                    row['is_coref_A'], row['is_coref_B'])

            sample_namedtuple = Sample(**sample)
            self.samples.append(sample_namedtuple)

    def _create_multiple_choices(self, tokens: List[int],
                                 offsets: Dict[str, List[int]]) -> List[int]:
        CLS = [self.tokenizer.cls_token]
        SEP = [self.tokenizer.sep_token]

        pronoun = tokens[offsets[self.pronoun_tag][0]]
        A_entity = tokens[offsets[self.start_A_tag]
                          [0]:offsets[self.end_A_tag][0]]
        B_entity = tokens[offsets[self.start_B_tag]
                          [0]:offsets[self.end_B_tag][0]]

        final_tokens = []

        first = CLS + tokens + SEP + [pronoun, "is"] + A_entity + SEP
        second = CLS + tokens + SEP + [pronoun, "is"] + B_entity + SEP
        third = CLS + tokens + SEP + [pronoun, "is", "neither"] + SEP

        tokens_to_convert = [first, second, third]
        final_tokens = [self.tokenizer.convert_tokens_to_ids(
            choice) for choice in tokens_to_convert]

        return final_tokens

    def _get_offsets_list(self, offsets: Dict[str, List[int]]) -> List[int]:
        # 1 is added for the introduction of the CLS token
        offsets_A = [offsets[self.start_A_tag]
                     [0] + 1, offsets[self.end_A_tag][0] + 1]
        offsets_B = [offsets[self.start_B_tag]
                     [0] + 1, offsets[self.end_B_tag][0] + 1]

        return offsets_A + offsets_B + [offsets[self.pronoun_tag][0] + 1]

    def _insert_tag(self, text: str, offsets: Tuple[int, int],
                    start_tag: str, end_tag: str = None) -> str:
        start_off, end_off = offsets

        # Starting tag only
        if end_tag is None:
            text = text[:start_off] + start_tag + text[start_off:]
            return text

        text = text[:start_off] + start_tag + \
            text[start_off:end_off] + end_tag + text[end_off:]
        return text

    """
    The methods '_delimit_mentions' and '_tokenize' are inspired 
    by the work of `rakeshchada` in the repository 
    "https://github.com/rakeshchada/corefqa/blob/master/CorefSeq.ipynb"
    """

    def _delimit_mentions(self, row: pd.Series) -> str:
        text = row['text']
        pronoun = row['pron']
        A_entity = row['entity_A']
        B_entity = row['entity_B']

        # Sort the offsets in ascending order
        break_points = sorted([
            (self.pronoun_tag, row['p_offset']),
            (self.start_A_tag, row['offset_A']),
            (self.end_A_tag, row['offset_A'] + len(A_entity)),
            (self.start_B_tag, row['offset_B']),
            (self.end_B_tag, row['offset_B'] + len(B_entity)),
        ], key=lambda x: x[1])

        # When a new tag is inserted, the offset of the next tag
        # changes by the length of the inserted tag.
        len_added_tags = 0
        for tag, offset in break_points:
            offset += len_added_tags
            text = self._insert_tag(text, (offset, None), tag)
            len_added_tags += len(tag)

        # Truncate the text at the last tag inserted and append the pronoun at the end
        if self.truncate_up_to_pron:
            text = text[:offset+len(tag)] + pronoun

        return text

    def _tokenize(self, row: pd.Series) -> Tuple[List[int], Dict[str, List[int]]]:
        """
        Tokenize the text.
        If keep_tags is True, also the tags are tokenized.
        """

        tokens = []
        tag_labels = self.tag_labels
        offsets = {tag: [] for tag in tag_labels.values()}

        text = self._delimit_mentions(row)

        # Also the tags are added to the tokens
        if self.keep_tags:
            for token in self.tokenizer.tokenize(text):
                tokens.append(token)

                if token in [*tag_labels.values()]:
                    if "/" in token:  # End token
                        offsets[token].append(len(tokens)-1)
                    else:
                        offsets[token].append(len(tokens))

        # The tags are skipped
        else:
            for token in self.tokenizer.tokenize(text):
                if token in [*tag_labels.values()]:
                    offsets[token].append(len(tokens))
                    continue
                tokens.append(token)

        return tokens, offsets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class Collator:
    """
    Collator for Sequence Classification.

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

    def __init__(self, device: str, pad: int = 0,
                 truncate_len: int = 512, labeled=True):
        self.device = device
        self.pad = pad
        self.truncate_len = truncate_len
        self.labeled = labeled

    def __call__(self, batch: list) -> Dict[str, torch.Tensor]:

        if self.labeled:
            batch_features, batch_offsets, batch_labels = zip(*batch)

        else:
            batch_features, batch_offsets = zip(*batch)

        max_len = compute_max_len(batch_features, self.truncate_len)

        collate_sample = {}

        # Features
        padded_features = pad_sequence(batch_features, max_len, self.pad)
        features_tensor = torch.tensor(padded_features, device=self.device)
        collate_sample['features'] = features_tensor

        # Offsets
        offsets_tensor = torch.tensor(batch_offsets, device=self.device)
        collate_sample['offsets'] = offsets_tensor

        if not self.labeled:
            return collate_sample

        # Labels
        labels_tensor = torch.tensor(
            batch_labels, dtype=torch.uint8, device=self.device)
        collate_sample['labels'] = labels_tensor

        return collate_sample


class Collator_Multi_Choice:
    """
    Collator for Multiple Choice Classification.

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

    def __init__(self, device: str, pad: int = 0,
                 truncate_len: int = 512, labeled=True):
        self.device = device
        self.pad = pad
        self.truncate_len = truncate_len
        self.labeled = labeled

    def __call__(self, batch: list) -> Dict[str, torch.Tensor]:
        if self.labeled:
            batch_features, _, batch_labels = zip(*batch)

        else:
            batch_features, _ = zip(*batch)

        batch_size = len(batch_features)
        num_choices = len(batch_features[0])

        collate_sample = {}

        # Features
        flattened_batch_features = sum(batch_features, [])
        max_len = compute_max_len(flattened_batch_features, self.truncate_len)

        padded_flattened_features = pad_sequence(
            flattened_batch_features, max_len, self.pad)
        flattened_features_tensor = torch.tensor(
            padded_flattened_features, device=self.device)
        features_tensor = flattened_features_tensor.view(
            batch_size, num_choices, -1)

        collate_sample['features'] = features_tensor

        if not self.labeled:
            return collate_sample

        # Labels
        labels_tensor = torch.tensor(
            batch_labels, dtype=torch.uint8, device=self.device)
        collate_sample['labels'] = labels_tensor

        return collate_sample


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
