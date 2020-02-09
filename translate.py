import torch
import argparse
import os

import constants as Constants
import pandas as pd
import dill as pickle

from transformer import Transformer
from translator import Translator
from dataset import TranslationDataset
from tqdm import tqdm


def read_instances_from_csv(file_path):
    df = pd.read_csv(file_path)

    src_word_insts = []
    trg_word_insts = []

    for row in df.iterrows():
        src_sent = row[1]["src"].strip().split()
        trg_sent = row[1]["trg"].strip().split()

        src_word_insts += [[Constants.BOS_WORD] + src_sent + [Constants.EOS_WORD]]
        trg_word_insts += [[Constants.BOS_WORD] + trg_sent + [Constants.EOS_WORD]]

    return src_word_insts, trg_word_insts


def convert_instance_to_idx_seq(word_insts, word2idx):
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in word_insts]


def load_model(opt, device):
    checkpoint = torch.load(opt.model_path, map_location=device)
    model_opt = checkpoint["settings"]
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
        dropout=model_opt.dropout,
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    print("[Info] Trained model state loaded.")
    return model


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-model_path", type=str, required=True)
    parser.add_argument("-test_path", type=str, default="data/five_test.csv")
    parser.add_argument("-vocab_path", type=str, default="preprocessed/dataset")
    parser.add_argument("-save_dir", type=str, default="test")
    parser.add_argument("-beam_size", type=int, default=5)
    parser.add_argument("-max_seq_len", type=int, default=10)
    parser.add_argument("-no_cuda", action="store_true")

    opt = parser.parse_args()

    return opt


def main():
    opt = get_args()

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    device = torch.device("cuda") if not opt.no_cuda else torch.device("cpu")

    preprocess_data = torch.load(opt.vocab_path)
    preprocess_settings = preprocess_data["settings"]
    test_src_word_insts, test_trg_word_insts = read_instances_from_csv(opt.test_path)
    test_src_idx_insts = convert_instance_to_idx_seq(
        test_src_word_insts, preprocess_data["dict"]["src"]
    )

    transformer = load_model(opt, device)
    translator = Translator(
        transformer,
        opt.beam_size,
        opt.max_seq_len,
        Constants.PAD,
        Constants.PAD,
        Constants.BOS,
        Constants.EOS,
    ).to(device)

    src_idx2word = {idx: word for word, idx in preprocess_data["dict"]["src"].items()}
    trg_idx2word = {idx: word for word, idx in preprocess_data["dict"]["tgt"].items()}

    pred_dict = {"src": [], "gold": []}
    pred_names = ["pred_" + str(idx) for idx in range(1, opt.beam_size + 1)]
    for pred_name in pred_names:
        pred_dict[pred_name] = []

    dec_attn_list = []
    enc_dec_attn_list = []
    for src_seq, gold_seq in tqdm(
        zip(test_src_idx_insts, test_trg_word_insts),
        mininterval=2,
        desc="  - (Test)",
        leave=False,
    ):
        src_word_seq = [src_idx2word.get(idx, Constants.UNK_WORD) for idx in src_seq]
        pred_dict["src"].append("".join(src_word_seq[1:-1]))
        pred_dict["gold"].append("".join(gold_seq[1:-1]))
        # Return all beam
        (
            all_pred_seq,
            all_dec_attn_list,
            all_enc_dec_attn_list,
        ) = translator.translate_sentence(
            torch.LongTensor([src_seq]).to(device), rtn_all_beam=True
        )
        for idx, pred_seq in enumerate(all_pred_seq):
            pred_word_seq = "".join(
                [trg_idx2word.get(idx, Constants.UNK_WORD) for idx in pred_seq][1:-1]
            )
            pred_dict[pred_names[idx]].append(pred_word_seq)
        dec_attn = all_dec_attn_list[-1]
        dec_attn_list.append(dec_attn)
        enc_dec_attn = all_enc_dec_attn_list[-1]
        enc_dec_attn_list.append(enc_dec_attn)

    with open(os.path.join(opt.save_dir, "dec_attn_list.pkl"), "wb") as dec_attn_f:
        pickle.dump(dec_attn_list, dec_attn_f)
    with open(
        os.path.join(opt.save_dir, "enc_dec_attn_list.pkl"), "wb"
    ) as enc_dec_attn_f:
        pickle.dump(enc_dec_attn_list, enc_dec_attn_f)

    pred_df = pd.DataFrame(pred_dict)
    pred_df.to_csv(os.path.join(opt.save_dir, "pred.csv"))


if __name__ == "__main__":
    main()
