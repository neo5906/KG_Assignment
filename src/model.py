# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, pretrained_embeddings=None,
                 embedding_dim=300, hidden_dim=256, num_layers=2, dropout=0.5):
        super(BiLSTMCRF, self).__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True, padding_idx=0)
            embedding_dim = pretrained_embeddings.shape[1]
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=num_layers, bidirectional=True,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, x, lengths, tags=None):
        embeds = self.embedding(x)
        embeds = self.dropout(embeds)
        packed_embeds = nn.utils.rnn.pack_padded_sequence(embeds, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_lstm_out, _ = self.lstm(packed_embeds)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_lstm_out, batch_first=True)
        emissions = self.hidden2tag(lstm_out)

        # 对齐长度
        batch_size, actual_len, _ = emissions.shape
        max_len = x.shape[1]
        if actual_len < max_len:
            padding = torch.zeros(batch_size, max_len - actual_len, emissions.shape[-1]).to(emissions.device)
            emissions = torch.cat([emissions, padding], dim=1)
        elif actual_len > max_len:
            emissions = emissions[:, :max_len, :]

        mask = (x != 0)
        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            pred_tags = self.crf.decode(emissions, mask=mask)
            return pred_tags