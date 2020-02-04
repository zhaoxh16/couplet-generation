
# %%
import torch

import constants as Constants

from transformer import Transformer
from translator import Translator


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model_opt = checkpoint['settings']
    print(model_opt)

    model = Transformer(
        model_opt.src_vocab_size,
        model_opt.trg_vocab_size,
        d_model=model_opt.d_model,
        d_inner_hid=model_opt.d_inner_hid,
        d_k=model_opt.d_k,
        d_v=model_opt.d_v,
        n_head=model_opt.n_head,
        n_layers=model_opt.n_layers,
        dropout=model_opt.dropout
    ).to(device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model


device = torch.device('cuda')
model = load_model("test/trained.chkpt", device)

# %%
model.eval()

# %%
src_seq = [2, 2243, 3050, 2188, 186, 829, 3]
trg_seq = [2, 2389, 1352, 936, 5125, 55, 3]
pred_seq, enc_attn, dec_attn, enc_dec_attn = model(torch.LongTensor([src_seq]).to(device), torch.LongTensor([trg_seq[:-1]]).to(device), rtn_enc_dec_attn=True)

# %%
print(pred_seq.size())

# %%
print(len(enc_dec_attn))