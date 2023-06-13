#!/usr/bin/env python3
# Copyright (c) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT
#
import os
import pdb

import nltk
import numpy as np
import torch

import utils


class GPT2:
    # https://github.com/huggingface/transformers/issues/1458
    def __init__(self):
        super(GPT2, self).__init__()
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to("cuda")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.word_dim = 768

    def embeds(self, word_tk):
        tkidx = self.tokenizer.encode(word_tk, add_prefix_space=True)
        emb = self.model.transformer.wte.weight[tkidx, :]
        return emb  # .numpy()

    def get_word_dim(self):
        return self.word_dim

    def word_embed(self, sentence):
        with torch.no_grad():
            tokens = nltk.tokenize.word_tokenize(sentence.lower())
            word_feats = torch.row_stack([self.embeds(tk) for tk in tokens])
        return word_feats


class BERT:
    # https://huggingface.co/docs/transformers/model_doc/bert
    def __init__(self):
        super(BERT, self).__init__()
        from transformers import BertModel, BertTokenizer

        self.model = BertModel.from_pretrained("bert-base-uncased").to("cuda")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.word_dim = 768

    def get_word_dim(self):
        return self.word_dim

    def word_embed(self, sentence):
        with torch.no_grad():
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True).to("cuda")
            outputs = self.model(**inputs)
            word_feats = outputs.last_hidden_state
        return torch.tensor(word_feats.squeeze()).cuda()


class GloVe:
    def __init__(self):
        super(GloVe, self).__init__()
        import torchtext

        self.model = torchtext.vocab.GloVe(name="6B", dim=300)
        self.word_dim = 300

    def get_word_dim(self):
        return self.word_dim

    def word_embed(self, sentence):
        tokens = nltk.tokenize.word_tokenize(sentence.lower())
        word_feats = np.row_stack([self.model[tk] for tk in tokens])
        return torch.tensor(word_feats).cuda()


def globals_init(args):
    global puzzle_diff, puzzle_diff_str, osp, rand, MAX_VAL, MAX_DECODE_STEPS, max_qlen
    global num_puzzles, seed, icon_class_ids, signs
    global SEQ_PUZZLES, NUM_CLASSES_PER_PUZZLE, device, SMART_DATASET_INFO_FILE
    global word_dim, word_embed
    global puzzles_not_included, num_actual_puzz
    global PS_VAL_IDX, PS_TEST_IDX

    device = "cuda"
    puzzle_diff = {"easy": ""}  # {'easy': 'e', 'medium': 'm', 'hard': 'h'}
    puzzle_diff_str = {"easy": ""}
    osp = os.path.join
    rand = lambda: np.random.rand() > 0.5
    MAX_VAL = 0
    MAX_DECODE_STEPS = 10  # number of steps to decode the LSTM.
    num_puzzles = 101
    max_qlen = 110
    seed = 10
    icon_dataset_path = "./dataset/icon-classes.txt"  #'/homes/cherian/train_data/NAR/SMART/SMART_cpl/puzzles/anoops/resources/icons-50/Icons-50/'
    icon_class_ids = utils.get_icon_dataset_classes(icon_dataset_path)  # os.listdir(icon_dataset_path) # puzzle 1
    signs = np.array(["+", "-", "x", "/"])  # puzzle 58
    NUM_CLASSES_PER_PUZZLE = {}
    SEQ_PUZZLES = [16, 18, 35, 39, 63, 100]
    SMART_DATASET_INFO_FILE = "./dataset/SMART_info_v2.csv"
    num_actual_puzz = 102
    puzzles_not_included = set([])
    PS_VAL_IDX = [7, 43, 64]
    PS_TEST_IDX = [94, 95, 96, 97, 98, 99, 101, 61, 62, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77]

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    # if gpt2
    if args.word_embed == "glove":
        Embed = GloVe()
        word_dim = Embed.get_word_dim()
        word_embed = Embed.word_embed
    elif args.word_embed == "gpt":
        Embed = GPT2()
        word_dim = Embed.get_word_dim()
        word_embed = Embed.word_embed
    elif args.word_embed == "bert":
        Embed = BERT()
        word_dim = Embed.get_word_dim()
        word_embed = Embed.word_embed
    else:
        print("word embedding used is %s" % (args.word_embed))
