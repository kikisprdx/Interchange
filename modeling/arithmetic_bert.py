import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class ArithmeticBertModule(nn.Module):
    def __init__(self, num_labels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_labels = num_labels
        self.dropout_prob = dropout

        # load pretrained BERT and tokenizer from HuggingFace
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        hidden_size = self.bert.config.hidden_size  # 768 for bert-base

        # classification head: drop some activations to reduce overfitting,
        # then project from 768 dimensions down to num_labels (3)
        self.dropout = nn.Dropout(dropout)
        # named 'logits' so the compgraph in compgraphs/arithmetic_bert.py can
        # call self.model.logits(x) directly
        self.logits = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Args:
            input_ids:      (batch, seq_len)  — token ids from tokenizer
            attention_mask: (batch, seq_len)  — 1 for real tokens, 0 for padding

        Returns:
            logits:        (batch, num_labels)         — raw class scores
            hidden_states: tuple of 13 tensors,
                           each shape (batch, seq_len, 768)
                           index 0 = embedding layer output
                           index 1-12 = transformer layer outputs
        """
        # run BERT — ask it to return hidden states from every layer
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # gives us all 13 layers
        )

        # hidden_states is a tuple of 13 tensors: (batch, seq_len, 768)
        hidden_states = outputs.hidden_states

        # pooler_output = Linear(768,768) + Tanh applied to [CLS] token
        # this matches what compgraphs/arithmetic_bert.py does: embed → layers → pool → logits
        cls_output = outputs.pooler_output

        # dropout + linear projection → (batch, num_labels)
        logits = self.logits(self.dropout(cls_output))

        return logits, hidden_states

    @property
    def device(self) -> torch.device:
        # compgraph needs to know which device the model is on
        return next(self.parameters()).device

    def config(self):
        """Returns the model's hyperparameters as a plain dict."""
        return {
            "num_labels": self.num_labels,
            "dropout": self.dropout_prob,
            "hidden_size": self.bert.config.hidden_size,
            "bert_model": "bert-base-uncased",
        }
