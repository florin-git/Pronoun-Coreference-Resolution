import re

import numpy as np
from typing import List, Tuple, Dict

from model import Model


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
    return RandomBaseline(False, False)


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
            entities if entities is not None else self.predict_entities(entities, toks)
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

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def predict(
        self, sentences: List[Dict]
    ) -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        pass
