#!/usr/bin/env python3
# Copyright (c) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT
#
import nltk

# make sure nltk works fine.
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print("downloading nltk/punkt tokenizer")
    nltk.download("punkt")

import argparse
import glob
import os
import pickle
from collections import Counter

from utils import save_file


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(text_rows, threshold):
    """Build a simple vocabulary wrapper."""

    print("total QA pairs", len(text_rows))
    counter = Counter()

    for text in text_rows:
        tokens = nltk.tokenize.word_tokenize(text.lower())
        counter.update(tokens)

    counter = sorted(counter.items(), key=lambda item: item[1], reverse=True)
    save_file(dict(counter), "dataset/VideoQA/word_count.json")
    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [item[0] for item in counter if item[1] >= threshold]
    print(len(words))
    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word("<pad>")
    vocab.add_word("<start>")
    vocab.add_word("<end>")
    vocab.add_word("<unk>")

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)

    return vocab


def read_csv(csvfilename):
    import csv

    qa_info = []
    with open(csvfilename, newline="") as csvfile:
        datareader = csv.DictReader(csvfile)
        for row in datareader:
            qa_info.append(row["Question"])
            qa_info.append(row["A"] + " " + row["B"] + " " + row["C"] + " " + row["D"] + " " + row["E"])
    return qa_info


def process_text_for_puzzle(args):
    vocab_path = os.path.join(args.save_root, "vocab_puzzle_" + args.puzzle_ids_str + ".pkl")
    if os.path.exists(vocab_path):
        print("loading vocab %s" % (vocab_path))
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
    else:
        text_rows = []
        for puzzle_id in args.puzzle_ids:
            print("reading puzzle %s" % (puzzle_id))
            text_files = glob.glob(os.path.join(args.data_root, str(puzzle_id), "puzzle_%s.csv" % (puzzle_id)))
            for t in range(len(text_files)):
                rows = read_csv(text_files[t])
                text_rows = text_rows + rows
        vocab = build_vocab(text_rows, threshold=3)
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)
        print("generating new vocab for %s: num_words=%d" % (args.puzzle_ids_str, len(vocab)))
    return vocab


def main(args):
    vocab = build_vocab(args.caption_path, args.threshold)
    vocab_path = args.vocab_path
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--anno_path", type=str, default="dataset/nextqa/train.csv", help="path for train annotation file"
    )
    parser.add_argument(
        "--vocab_path", type=str, default="dataset/VideoQA/vocab.pkl", help="path for saving vocabulary wrapper"
    )
    parser.add_argument("--threshold", type=int, default=1, help="minimum word count threshold")
    args = parser.parse_args()
    main(args)
