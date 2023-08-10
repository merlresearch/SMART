#!/usr/bin/env python3
# Copyright (c) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT
#

import warnings

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
import pdb
import pickle

import clip
import numpy as np
import torch.nn.functional as F
from PIL import Image

import globvars as gv


# Vision and Language pretrained models.
class SMART_VL_CLIP_Net(nn.Module):
    def __init__(self, args, VL_backbone):
        super(SMART_VL_CLIP_Net, self).__init__()
        vocab_path = args.vocab_path
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)

        self.num_opts = 5
        self.out_dim = args.feat_size
        self.h_sz = 256
        self.feat_size = 512
        self.dummy_question = None
        self.model_name = args.model_name
        self.use_clip_text = args.use_clip_text
        self.loss_type = args.loss_type
        self.monolithic = args.monolithic
        self.use_single_image_head = args.use_single_image_head
        self.train_backbone = args.train_backbone
        self.sorted_puzzle_ids = np.sort(np.array([int(ii) for ii in args.puzzle_ids]))

        if args.loss_type == "classifier" or args.loss_type == "puzzle_tails":
            self.max_val = gv.MAX_VAL + 1
        elif args.loss_type == "regression":
            self.max_val = 1

        self.preprocess = args.preprocess
        self.VL_backbone = VL_backbone
        self.create_puzzle_head(args)

        self.q_MLP = nn.Sequential(
            nn.Linear(self.feat_size, self.h_sz),
            nn.ReLU(),
            nn.Linear(self.h_sz, self.out_dim),
            nn.ReLU(),
        )

        self.qv_fusion = nn.Sequential(
            nn.Linear(self.out_dim * 2, self.out_dim),
            nn.ReLU(),
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(),
        )
        if self.monolithic:
            self.qvo_fusion = nn.Sequential(nn.Linear(self.out_dim, self.max_val))
        else:
            self.create_puzzle_tail(args)

    def create_puzzle_head(self, args):
        if args.use_single_image_head:
            self.im_encoder = nn.Sequential(
                nn.Linear(self.feat_size, self.out_dim), nn.ReLU(), nn.Linear(self.out_dim, self.out_dim)
            )
        else:
            self.puzzle_ids = args.puzzle_ids
            im_encoder = [nn.Sequential(nn.Linear(self.out_dim, 1))]
            for i in range(1, gv.num_puzzles + 1):
                im_encoder.append(
                    nn.Sequential(
                        nn.Linear(self.feat_size, self.out_dim), nn.ReLU(), nn.Linear(self.out_dim, self.out_dim)
                    )
                )
            self.im_encoder = nn.ModuleList(im_encoder)

    def create_puzzle_tail(self, args):
        self.puzzle_ids = args.puzzle_ids
        ans_decoder = [
            nn.Sequential(nn.Linear(self.out_dim, 1))
        ]  # start with a dummy as we are 1-indexed wrt puzzle ids.
        if args.puzzles == "all":
            puzzles = range(1, gv.num_puzzles + 1)
        else:
            puzzles = self.puzzle_ids
        for pid in puzzles:  # self.puzzle_ids:
            num_classes = gv.NUM_CLASSES_PER_PUZZLE[str(pid)] if args.loss_type == "classifier" else 1
            if int(pid) not in gv.SEQ_PUZZLES:
                ans_decoder.append(
                    nn.Sequential(
                        nn.Linear(self.out_dim, self.out_dim),
                        nn.ReLU(),
                        nn.Linear(self.out_dim, self.out_dim),
                        nn.ReLU(),
                        nn.Linear(self.out_dim, num_classes),
                    )
                )
            else:
                ans_decoder.append(nn.LSTM(self.out_dim, num_classes, num_layers=1, batch_first=True))
        self.ans_decoder = nn.ModuleList(ans_decoder)

    def process(self, im, q_text):
        q_text = self.decode_text(q_text)
        text = clip.tokenize(q_text, truncate=True).to("cuda")
        return im, text

    def encode_image(self, im_feat, pids=None):
        if self.use_single_image_head:
            y = self.im_encoder(im_feat)
        else:
            y = torch.zeros(len(im_feat), self.out_dim).cuda()
            for t in range(len(self.puzzle_ids)):
                idx = pids == int(self.puzzle_ids[t])
                idx = idx.cuda()
                if idx.sum() > 0:
                    y[idx] = F.relu(self.im_encoder[int(self.puzzle_ids[t])](im_feat[idx]))
        return y

    def encode_text(self, q_feat):
        x = F.relu(self.q_MLP(q_feat))
        return x

    def decode_image(self, im_list):
        """convert torch tensor images back to Image bcos VL FLAVA model works with images."""
        im_list = (im_list.permute(0, 2, 3, 1) * 255).cpu().numpy().astype("uint8")
        im_list = [Image.fromarray(im_list[ii]) for ii in range(len(im_list))]  # convert im
        return im_list

    def decode_text(self, text):
        get_range = lambda x: range(1, x) if x < 70 else range(x - 70 + 4, x)
        tt = text.cpu()
        text = [
            " ".join([self.vocab.idx2word[int(j)] for j in tt[i][get_range(torch.nonzero(tt[i])[-1])]])
            for i in range(len(tt))
        ]
        return text

    def seq_decoder(self, decoder, feat):
        """run the LSTM decoder sequentially for k steps"""
        out = [None] * gv.MAX_DECODE_STEPS
        hx = None
        for k in range(gv.MAX_DECODE_STEPS):
            try:
                out[k], hx = decoder(feat, hx)
            except:
                pdb.set_trace()
        return out

    def decode_individual_puzzles(self, feat, pids):
        upids = torch.unique(pids)
        out_feats = {}
        for t in range(len(upids)):
            idx = pids == upids[t]
            key = str(upids[t].item())
            key_idx = np.where(int(key) == np.array(self.sorted_puzzle_ids))[0][0] + 1  # +1 because we use 1-indexed.
            if upids[t] not in gv.SEQ_PUZZLES:
                out_feats[int(key)] = self.ans_decoder[key_idx](feat[idx])
            else:
                out_feats[int(key)] = self.seq_decoder(self.ans_decoder[key_idx], feat[idx])
        return out_feats

    def forward(self, im, q=None, puzzle_ids=None):
        im, text = self.process(im, q)

        if self.train_backbone:
            im_feat = self.VL_backbone.encode_image(im)
            q_feat = self.VL_backbone.encode_text(text)
        else:
            with torch.no_grad():
                im_feat = self.VL_backbone.encode_image(im)
                q_feat = self.VL_backbone.encode_text(text)

        im_feat = self.encode_image(im_feat.float(), puzzle_ids)
        q_feat = self.encode_text(q_feat.float())
        qv_feat = self.qv_fusion(torch.cat([im_feat, q_feat], dim=1))

        if self.monolithic:
            qv_feat = qv_feat.unsqueeze(1)
            qvo_feat = self.qvo_fusion(qv_feat).squeeze()
        else:
            qvo_feat = self.decode_individual_puzzles(qv_feat, puzzle_ids)

        return qvo_feat
