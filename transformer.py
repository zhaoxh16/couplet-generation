import torch
import torch.nn as nn
import numpy as np

import constants as Constants

from torch.nn.modules import Transformer as TransformerPart


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()
    

class Transformer(nn.Module):
    def __init__(self, n_src_vocab, n_trg_vocab, d_model=512, d_inner_hid=2048, d_k=64, d_v=64, 
                 n_head=8, n_layers=6, dropout=0.1):
        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_model, padding_idx=Constants.PAD)
        self.src_position_enc = PositionalEncoding(d_model)
        self.src_emb_dropout = nn.Dropout(p=dropout)

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_model, padding_idx=Constants.PAD)
        self.trg_position_enc = PositionalEncoding(d_model)
        self.trg_emb_dropout = nn.Dropout(p=dropout)

        self.transformer_part = TransformerPart(d_model=d_model, nhead=n_head, num_encoder_layers=n_layers,
                                                num_decoder_layers=n_layers, dim_feedforward=d_inner_hid, 
                                                dropout=dropout)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def src_embedding_and_pos_encoding(self, src_seq):
        return torch.transpose(self.src_emb_dropout(self.src_position_enc(self.src_word_emb(src_seq))), 1, 0)
    
    def trg_embedding_and_pos_encoding(self, trg_seq):
        return torch.transpose(self.trg_emb_dropout(self.trg_position_enc(self.trg_word_emb(trg_seq))), 1, 0)

    def forward(self, src_seq, trg_seq, trg_mask):
        """
        src_seq: :math:`(batch_size, src_seq_len)`
        trg_seq: :math:`(batch_size, trg_seq_len)`
        trg_mask: :math:`(trg_seq_len, trg_seq_len)`
        目前网络仅支持所有句子长度都相等，没有padding的情况
        """
        src_enc_output = self.src_embedding_and_pos_encoding(src_seq)  # src_enc_output: `(src_seq_len, batch_size, d_model)`
        trg_enc_output = self.trg_embedding_and_pos_encoding(trg_seq)  # trg_enc_output: `(trg_seq_len, batch_size, d_model)`

        transformer_part_output = self.transformer_part(src_enc_output, trg_enc_output, tgt_mask=trg_mask)
        seq_logit = torch.transpose(self.trg_word_prj(transformer_part_output), 1, 0)
        return seq_logit.contiguous().view(-1, seq_logit.size(2))
