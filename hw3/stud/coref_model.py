from transformers import (
    BertModel,
    logging
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import torch
from torch import nn
from stud.arguments import *


class Entity_Resolution_Head(nn.Module):
    def __init__(self, bert_hidden_size: int, args: ModelArguments):
        super().__init__()

        self.args = args
        self.bert_hidden_size = bert_hidden_size

        input_size_pronoun = bert_hidden_size
        input_size_entities = bert_hidden_size * 6
        if self.args.output_strategy == "concat":
            input_size_pronoun *= 4
            input_size_entities *= 4

        self.ffnn_pronoun = nn.Sequential(
            nn.Linear(input_size_pronoun, bert_hidden_size),
            nn.LayerNorm(bert_hidden_size),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(bert_hidden_size, args.head_hidden_size),
        )

        self.ffnn_entities = nn.Sequential(
            nn.Linear(input_size_entities, bert_hidden_size),
            nn.LayerNorm(bert_hidden_size),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(bert_hidden_size, args.head_hidden_size),
        )

        linear_hidden_size = args.linear_hidden_size

        # self.bilinear = torch.nn.Bilinear(args.head_hidden_size, args.head_hidden_size, bilinear_hidden_size, bias=False)
        self.linear = torch.nn.Linear(
            args.head_hidden_size * 2, linear_hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(args.dropout)

        self.classifier = nn.Linear(linear_hidden_size, args.num_output)

    def forward(self, bert_outputs, offsets):

        pron_emb, ent_emb = self._retrieve_pron_and_ent_embeddings(
            bert_outputs, offsets)

        x_pron = self.ffnn_pronoun(pron_emb)
        x_ent = self.ffnn_entities(ent_emb)

        # x = self.bilinear(x_pron, x_ent)
        x = self.linear(torch.cat([x_pron, x_ent], dim=-1))
        x = self.activation(x)
        x = self.dropout(x)

        output = self.classifier(x)
        return output

    def _retrieve_pron_and_ent_embeddings(self, bert_embeddings: torch.Tensor,
                                          offsets: torch.Tensor):
        embeddings_A = []
        embeddings_B = []
        pronouns_embedding = []

        # Consider embeddings and offsets in each batch separately
        for embeddings, off in zip(bert_embeddings, offsets):
            start_A = off[0]
            end_A = off[1]
            start_B = off[2]
            end_B = off[3]
            pron_off = off[-1]

            emb_A = torch.cat([embeddings[start_A], embeddings[end_A-1],
                              embeddings[start_A:end_A].mean(dim=0)], dim=-1)
            embeddings_A.append(emb_A)

            emb_B = torch.cat([embeddings[start_B], embeddings[end_B-1],
                              embeddings[start_B:end_B].mean(dim=0)], dim=-1)
            embeddings_B.append(emb_B)

            pronouns_embedding.append(embeddings[pron_off])

        # batch_size x (embedding_dim * 3 * 2)
        merged_entities_embeddings = torch.cat([
            torch.stack(embeddings_A, dim=0),
            torch.stack(embeddings_B, dim=0),
        ], dim=1)

        # batch_size x embedding_dim
        stacked_pronouns_embedding = torch.stack(pronouns_embedding, dim=0)

        return stacked_pronouns_embedding, merged_entities_embeddings


class CR_Model(nn.Module):
    """The main model."""

    def __init__(self, device, bert_model: str, tokenizer: PreTrainedTokenizerBase, args: ModelArguments):
        super().__init__()
        logging.set_verbosity_error()

        self.args = args

        if bert_model in {"bert-base-uncased", "bert-base-cased"}:
            self.bert_hidden_size = 768
        elif bert_model in {"bert-large-uncased", "bert-large-cased"}:
            self.bert_hidden_size = 1024
        else:
            self.bert_hidden_size = args.bert_hidden_size

        self.bert = BertModel.from_pretrained(
            bert_model).to(device, non_blocking=True)

        # If the tag tokens (e.g., <p>, <a> etc.) are present in the features,
        # the embedding dimension of the bert embeddings must be changed
        # to be compliant with the new size of the tokenizer vocabulary.
        if args.resize_embeddings:
            self.bert.resize_token_embeddings(len(tokenizer.vocab))

        self.head = Entity_Resolution_Head(self.bert_hidden_size, args).to(
            device, non_blocking=True)

    def forward(self, sample):
        x = sample['features']
        x_offsets = sample['offsets']

        bert_outputs = self.bert(
            x, attention_mask=(x > 0).long(),
            token_type_ids=None, output_hidden_states=True)

        if self.args.output_strategy == "last":
            out = bert_outputs.last_hidden_state

        elif self.args.output_strategy == "concat":
            out = torch.cat([bert_outputs.hidden_states[x]
                            for x in [-1, -2, -3, -4]], dim=-1)

        elif self.args.output_strategy == "sum":
            layers_to_sum = torch.stack(
                [bert_outputs.hidden_states[x] for x in [-1, -2, -3, -4]], dim=0)
            out = torch.sum(layers_to_sum, dim=0)

        else:
            raise ValueError("Unsupported output strategy.")

        head_outputs = self.head(out, x_offsets)

        return head_outputs
