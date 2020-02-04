import torch
import argparse
import random
import math
import time
import os

import numpy as np
import constants as Constants
import torch.nn.functional as F

from tqdm import tqdm
from transformer import Transformer
from optimizer import ScheduledOptim
from torch.optim import Adam
from dataset import TranslationDataset, paired_collate_fn, collate_fn


def prepare_dataloaders(data, opt):
    print(data["settings"])

    train_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data["dict"]["src"],
            tgt_word2idx=data["dict"]["tgt"],
            src_insts=data["train"]["src"],
            tgt_insts=data["train"]["tgt"],
        ),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data["dict"]["src"],
            tgt_word2idx=data["dict"]["tgt"],
            src_insts=data["valid"]["src"],
            tgt_insts=data["valid"]["tgt"],
        ),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
    )

    src_vocab_size = train_loader.dataset.src_vocab_size
    trg_vocab_size = train_loader.dataset.tgt_vocab_size

    return train_loader, valid_loader, src_vocab_size, trg_vocab_size


def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    """ Apply label smoothing if needed """

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    """ Calculate cross entropy loss, apply label smoothing if needed. """

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction="sum")
    return loss


def train_epoch(model, training_data, optimizer, device, smoothing=True):
    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = "  - (Training)   "
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):
        # prepare data
        src_seq, trg_seq = map(lambda x: x.to(device), batch)
        gold = trg_seq[:, 1:]
        trg_seq = trg_seq[:, :-1]
        # trg_mask = generate_square_subsequent_mask(trg_seq.size(1)).to(device)

        # forward
        optimizer.zero_grad()
        pred, enc_attn_list, dec_attn_list, enc_dec_attn_list = model(src_seq, trg_seq)

        # backward and update parameters
        loss, n_correct, n_word = cal_performance(
            pred, gold, Constants.PAD, smoothing=smoothing
        )
        loss.backward()
        optimizer.step_and_update_lr()

        # note keeping
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, validation_data, device):
    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = "  - (Validation) "
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):
            # prepare data
            src_seq, trg_seq = map(lambda x: x.to(device), batch)
            gold = trg_seq[:, 1:]
            trg_seq = trg_seq[:, :-1]
            # trg_mask = generate_square_subsequent_mask(trg_seq.size(1)).to(device)

            # forward
            pred, enc_attn_list, dec_attn_list, enc_dec_attn_list = model(
                src_seq, trg_seq
            )
            loss, n_correct, n_word = cal_performance(
                pred, gold, Constants.PAD, smoothing=False
            )

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy


def train(model, training_data, validation_data, optimizer, device, opt):
    def print_performances(header, loss, accu, start_time):
        print(
            "  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, "
            "elapse: {elapse:3.3f} min".format(
                header=f"({header})",
                ppl=math.exp(min(loss, 100)),
                accu=100 * accu,
                elapse=(time.time() - start_time) / 60,
            )
        )

    valid_losses = []
    for epoch_i in range(opt.epoch):
        print("[ Epoch", epoch_i, "]")

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, device, smoothing=opt.label_smoothing
        )
        print_performances("Training", train_loss, train_accu, start)

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device)
        print_performances("Validation", valid_loss, valid_accu, start)

        valid_losses += [valid_loss]

        checkpoint = {"epoch": epoch_i, "settings": opt, "model": model.state_dict()}

        if opt.save_model:
            if opt.save_mode == "all":
                model_name = (
                    opt.save_model
                    + "_epoch_"
                    + str(epoch_i)
                    + "_accu_{accu:3.3f}.chkpt".format(accu=100 * valid_accu)
                )
                torch.save(checkpoint, os.path.join(opt.save_dir, model_name))
            elif opt.save_mode == "best":
                model_name = opt.save_model + ".chkpt"
                if valid_loss <= min(valid_losses):
                    torch.save(checkpoint, os.path.join(opt.save_dir, model_name))
                    print("    - [Info] The checkpoint file has been updated.")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-data_path", type=str, default="preprocessed/dataset")
    parser.add_argument("-save_model", default="trained")
    parser.add_argument("-save_mode", type=str, default="all")
    parser.add_argument("-save_dir", type=str, default="testtt")
    parser.add_argument("-no_cuda", action="store_true")

    parser.add_argument("-epoch", type=int, default=100)
    parser.add_argument("-batch_size", type=int, default=1024)
    parser.add_argument("-random_seed", type=int, default=351)

    parser.add_argument("-l_rate", type=float, default=1e-3)
    parser.add_argument("-dropout", type=float, default=0.1)
    parser.add_argument("-n_warmup_steps", type=int, default=8000)
    parser.add_argument("-label_smoothing", action="store_true")

    parser.add_argument("-d_model", type=int, default=512)
    parser.add_argument("-d_inner_hid", type=int, default=2048)
    parser.add_argument("-d_k", type=int, default=64)
    parser.add_argument("-d_v", type=int, default=64)
    parser.add_argument("-n_head", type=int, default=8)
    parser.add_argument("-n_layers", type=int, default=6)

    opt = parser.parse_args()

    return opt


def main():
    opt = get_args()

    device = torch.device("cuda") if not opt.no_cuda else torch.device("cpu")

    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    else:
        print("[Warning] Save directory already exists.")

    data = torch.load(opt.data_path)
    (
        training_data,
        validation_data,
        src_vocab_size,
        trg_vocab_size,
    ) = prepare_dataloaders(data, opt)
    opt.src_vocab_size = src_vocab_size  # for save
    opt.trg_vocab_size = trg_vocab_size  # for save

    transformer = Transformer(
        src_vocab_size,
        trg_vocab_size,
        opt.d_model,
        opt.d_inner_hid,
        opt.d_k,
        opt.d_v,
        opt.n_head,
        opt.n_layers,
        opt.dropout,
    ).to(device)

    optimizer = ScheduledOptim(
        Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        1.0,
        opt.d_model,
        opt.n_warmup_steps,
    )

    train(transformer, training_data, validation_data, optimizer, device, opt)


if __name__ == "__main__":
    main()
