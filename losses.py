#!/usr/bin/env python3
# Copyright (c) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT
#
import torch.nn as nn

import globvars as gv


class Criterion(nn.Module):
    def __init__(self, args):
        super(Criterion, self).__init__()
        self.monolithic = args.monolithic  # just one classifier
        self.loss_type = args.loss_type
        if args.loss_type == "classifier":
            self.criterion = nn.CrossEntropyLoss()
        elif args.loss_type == "regression":
            self.criterion = nn.L1Loss()

    def compute_loss(self, a, b, pids):
        if self.monolithic:
            loss = self.criterion(a, b[:, 0])
        else:
            loss = 0
            for key in a.keys():
                idx = pids == int(key)
                if int(key) not in gv.SEQ_PUZZLES:
                    loss += self.criterion(
                        a[key], b[idx, 0]
                    )  # risky if idx and key entries are not matched. but then we will encouter an exception.
                else:
                    seq_loss = 0
                    for i in range(len(a[key])):
                        seq_loss += self.criterion(a[key][i], b[idx, i])  # .long()
                    seq_loss /= len(a[key])
                    loss += seq_loss
            loss = loss / len(a.keys())
        return loss

    def forward(self, a, b, pids=None):
        if self.loss_type == "classifier":
            loss = self.compute_loss(a, b.long(), pids)
        elif self.loss_type == "regression":
            loss = self.compute_loss(a, b.float(), pids)
        else:
            raise "Unknown loss type: use classifer/regression"
        return loss
