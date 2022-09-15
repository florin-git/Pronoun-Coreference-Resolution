import re
import yaml

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

from model import Model

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from stud.coref_model import *
from stud.arguments import *
from stud.gap_dataset import *
from stud.gap_utils import *


def build_model_123(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 1, 2 and 3 of the Coreference resolution pipeline.
            1: Ambiguous pronoun identification.
            2: Entity identification
            3: Coreference resolution
    """
    return RandomBaseline(True, True)


def build_model_23(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 2 and 3 of the Coreference resolution pipeline.
            2: Entity identification
            3: Coreference resolution
    """
    return RandomBaseline(False, True)


def build_model_3(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements step 3 of the Coreference resolution pipeline.
            3: Coreference resolution
    """
    return StudentModel(device)


class RandomBaseline(Model):
    def __init__(self, predict_pronoun: bool, predict_entities: bool):
        self.pronouns_weights = {
            "his": 904,
            "her": 773,
            "he": 610,
            "she": 555,
            "him": 157,
        }
        self.predict_pronoun = predict_pronoun
        self.pred_entities = predict_entities

    def predict(
        self, sentences: List[Dict]
    ) -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:
        predictions = []
        for sent in sentences:
            text = sent["text"]
            toks = re.sub("[.,'`()]", " ", text).split(" ")
            if self.predict_pronoun:
                prons = [
                    tok.lower() for tok in toks if tok.lower() in self.pronouns_weights
                ]
                if prons:
                    pron = np.random.choice(prons, 1, self.pronouns_weights)[0]
                    pron_offset = text.lower().index(pron)
                    if self.pred_entities:
                        entity = self.predict_entity(
                            predictions, pron, pron_offset, text, toks
                        )
                    else:
                        entities = [sent["entity_A"], sent["entity_B"]]
                        entity = self.predict_entity(
                            predictions, pron, pron_offset, text, toks, entities
                        )
                    predictions.append(((pron, pron_offset), entity))
                else:
                    predictions.append(((), ()))
            else:
                pron = sent["pron"]
                pron_offset = sent["p_offset"]
                if self.pred_entities:
                    entity = self.predict_entity(
                        predictions, pron, pron_offset, text, toks
                    )
                else:
                    entities = [
                        (sent["entity_A"], sent["offset_A"]),
                        (sent["entity_B"], sent["offset_B"]),
                    ]
                    entity = self.predict_entity(
                        predictions, pron, pron_offset, text, toks, entities
                    )
                predictions.append(((pron, pron_offset), entity))
        return predictions

    def predict_entity(self, predictions, pron, pron_offset, text, toks, entities=None):
        entities = (
            entities if entities is not None else self.predict_entities(
                entities, toks)
        )
        entity_idx = np.random.choice([0, len(entities) - 1], 1)[0]
        return entities[entity_idx]

    def predict_entities(self, entities, toks):
        offset = 0
        entities = []
        for tok in toks:
            if tok != "" and tok[0].isupper():
                entities.append((tok, offset))
            offset += len(tok) + 1
        return entities


class StudentModel(Model):

    def __init__(self, device):

        self.device = device
        self.tag_labels = {
            "pronoun_tag": "<p>",
            "start_A_tag": "<a>",
            "end_A_tag": "</a>",
            "start_B_tag": "<b>",
            "end_B_tag": "</b>"
        }
        self.model = None
        self.tokenizer = None

    def predict(
        self, sentences: List[Dict]
    ) -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:

        if self.model is None:
            path = "model/checkpoints/MentionScore/large/large-mention_score_872.pth"

            yaml_file = "model/checkpoints/MentionScore/large/predict.yaml"
            # Read configuration file with all the necessary parameters
            with open(yaml_file) as file:
                config = yaml.safe_load(file)

            model_args = ModelArguments(**config['model_args'])
            model_name_or_path = model_args.model_name_or_path
            tokenizer_name_or_path = model_args.tokenizer
            if tokenizer_name_or_path is None:
                tokenizer_name_or_path = model_name_or_path

            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path, never_split=list(self.tag_labels.values()))
            self.tokenizer.add_tokens(
                list(self.tag_labels.values()), special_tokens=True)

            self.model = CR_Model(self.device, model_name_or_path, self.tokenizer, model_args).to(
                self.device, non_blocking=True)
            print("Model Init")
            self.model.load_state_dict(
                torch.load(path, map_location=self.device))
            print("Model Loaded")
            print("Predicting...")

        df = pd.DataFrame(sentences)
        dataset = GAP_Dataset(df, self.tokenizer, self.tag_labels,
                              truncate_up_to_pron=False, labeled=False, cleaned=False)
        collator = Collator(self.device, labeled=False)

        predictions = []

        self.model.eval()
        with torch.no_grad():
            dataloader = DataLoader(dataset, batch_size=1,
                                    collate_fn=collator, shuffle=False)

            for sample, sentence in zip(dataloader, sentences):
                predicted_label_id = self.model(sample).argmax(1).item()
                pred_entity, pred_entity_offset = get_entity_and_offset_from_id(
                    predicted_label_id, sentence)
                pron, pron_offset = sentence['pron'], sentence['p_offset']

                predictions.append(
                    ((pron, pron_offset), (pred_entity, pred_entity_offset)))

        return predictions
