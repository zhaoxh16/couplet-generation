''' Handling the data io '''
import argparse
import torch
import constants as Constants
import os
import pandas as pd


def read_instances_from_file(inst_file, max_sent_len, keep_case):
    ''' Convert file into word seq lists and vocab '''

    word_insts = []
    trimmed_sent_count = 0
    with open(inst_file) as f:
        for sent in f:
            if not keep_case:
                sent = sent.lower()
            words = sent.split()
            if len(words) > max_sent_len:
                trimmed_sent_count += 1
            word_inst = words[:max_sent_len]

            if word_inst:
                word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
            else:
                word_insts += [None]

    print('[Info] Get {} instances from {}'.format(len(word_insts), inst_file))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))

    return word_insts


def read_instances_from_csv(inst_file, max_sent_len, keep_case):
    src_word_insts = []
    tgt_word_insts = []
    trimmed_sent_count = 0
    df = pd.read_csv(inst_file)
    for row in df.iterrows():
        src_sent = row[1]['src'].strip().split()
        tgt_sent = row[1]['trg'].strip().split()
        if len(src_sent) > max_sent_len or len(tgt_sent) > max_sent_len:
            trimmed_sent_count +=1
        src_word_inst = src_sent[:max_sent_len]
        tgt_word_inst = tgt_sent[:max_sent_len]
        src_word_insts += [[Constants.BOS_WORD] + src_word_inst + [Constants.EOS_WORD]]
        tgt_word_insts += [[Constants.BOS_WORD] + tgt_word_inst + [Constants.EOS_WORD]]
    print('[Info] Get {} instances from {}'.format(len(src_word_insts), inst_file))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'.format(trimmed_sent_count, max_sent_len))
    return src_word_insts, tgt_word_insts



def build_vocab_idx(word_insts, min_word_count):
    ''' Trim vocab by number of occurence '''

    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    word2idx = {
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK}

    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count >= min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx


def convert_instance_to_idx_seq(word_insts, word2idx):
    ''' Mapping words to idx sequence. '''
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in word_insts]


def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', required=True)
    # parser.add_argument('-train_tgt', required=True)
    parser.add_argument('-valid_src', required=True)
    # parser.add_argument('-valid_tgt', required=True)
    parser.add_argument('-output_dir', required=True)
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=50)
    parser.add_argument('-min_word_count', type=int, default=5)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)

    opt = parser.parse_args()

    # examine output directory
    output_dir = opt.output_dir
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise ValueError("Output directiory ({}) already exists and is not empty.".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Training set
    train_src_word_insts, train_tgt_word_insts = read_instances_from_csv(opt.train_src, opt.max_word_seq_len, opt.keep_case)
    # train_src_word_insts = read_instances_from_file(
        # opt.train_src, opt.max_word_seq_len, opt.keep_case)
    # train_tgt_word_insts = read_instances_from_file(
        # opt.train_tgt, opt.max_word_seq_len, opt.keep_case)

    if len(train_src_word_insts) != len(train_tgt_word_insts):
        print('[Warning] The training instance count is not equal.')
        min_inst_count = min(len(train_src_word_insts), len(train_tgt_word_insts))
        train_src_word_insts = train_src_word_insts[:min_inst_count]
        train_tgt_word_insts = train_tgt_word_insts[:min_inst_count]

    # - Remove empty instances
    train_src_word_insts, train_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(train_src_word_insts, train_tgt_word_insts) if s and t]))

    # Validation set
    valid_src_word_insts, valid_tgt_word_insts = read_instances_from_csv(opt.valid_src, opt.max_word_seq_len, opt.keep_case)
    # valid_src_word_insts = read_instances_from_file(
        # opt.valid_src, opt.max_word_seq_len, opt.keep_case)
    # valid_tgt_word_insts = read_instances_from_file(
        # opt.valid_tgt, opt.max_word_seq_len, opt.keep_case)

    if len(valid_src_word_insts) != len(valid_tgt_word_insts):
        print('[Warning] The validation instance count is not equal.')
        min_inst_count = min(len(valid_src_word_insts), len(valid_tgt_word_insts))
        valid_src_word_insts = valid_src_word_insts[:min_inst_count]
        valid_tgt_word_insts = valid_tgt_word_insts[:min_inst_count]

    # - Remove empty instances
    valid_src_word_insts, valid_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(valid_src_word_insts, valid_tgt_word_insts) if s and t]))

    # Build vocabulary
    if opt.vocab:
        predefined_data = torch.load(opt.vocab)
        assert 'dict' in predefined_data

        print('[Info] Pre-defined vocabulary found.')
        src_word2idx = predefined_data['dict']['src']
        tgt_word2idx = predefined_data['dict']['tgt']
    else:
        if opt.share_vocab:
            print('[Info] Build shared vocabulary for source and target.')
            word2idx = build_vocab_idx(
                train_src_word_insts + train_tgt_word_insts, opt.min_word_count)
            src_word2idx = tgt_word2idx = word2idx
        else:
            print('[Info] Build vocabulary for source.')
            src_word2idx = build_vocab_idx(train_src_word_insts, opt.min_word_count)
            print('[Info] Build vocabulary for target.')
            tgt_word2idx = build_vocab_idx(train_tgt_word_insts, opt.min_word_count)

    # word to index
    print('[Info] Convert source word instances into sequences of word index.')
    train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
    valid_src_insts = convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx)

    print('[Info] Convert target word instances into sequences of word index.')
    train_tgt_insts = convert_instance_to_idx_seq(train_tgt_word_insts, tgt_word2idx)
    valid_tgt_insts = convert_instance_to_idx_seq(valid_tgt_word_insts, tgt_word2idx)

    opt.max_token_seq_len = opt.max_word_seq_len + 2
    data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts},
        'valid': {
            'src': valid_src_insts,
            'tgt': valid_tgt_insts}}

    save_data_file = os.path.join(opt.output_dir, 'dataset')
    print('[Info] Dumping the processed data to pickle file', save_data_file)
    torch.save(data, save_data_file)
    opt_dict = vars(opt)
    with open(os.path.join(opt.output_dir, 'arguments.txt'), 'w') as writer:
        for key in opt_dict.keys():
            writer.write("%s = %s\n" % (key, str(opt_dict[key])))
    print('[Info] Finish.')


if __name__ == '__main__':
    main()
