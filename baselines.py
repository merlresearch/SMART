#!/usr/bin/env python3
# Copyright (c) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT
#
import os

import matplotlib.pyplot as plt
import numpy as np

import globvars as gv
import utils


def plot_baseline_perf(args, baseline_key, key, tot, split, suffix):
    plt.figure(figsize=(30, 4))
    ax = plt.gca()
    bpid = np.array(baseline_key)
    bclassids = np.arange(gv.NUM_CLASSES_PER_PUZZLE[key])
    x = np.histogram(bpid, np.arange(gv.NUM_CLASSES_PER_PUZZLE[key] + 1))[0]
    x = x / x.sum()
    ax.bar(bclassids, x)
    ax.set_xticks(bclassids)
    plt.savefig(
        "%s/stats/ans_distr/%s/%s/ans_distr_%s_%d_%s_%s.png" % (args.save_root, suffix, split, key, tot, split, suffix)
    )
    plt.close()
    return x


def get_baseline_performance(args, qa_info, split, tot, log=False):

    topK = lambda x: np.sort(x)[-2:].sum() / 2.0
    baseline = {}
    baseline_opts = {}
    for t in range(len(qa_info)):
        pid = qa_info[t]["puzzle_id"]
        if int(pid) not in gv.SEQ_PUZZLES and int(pid) != 58:
            if pid not in baseline:
                baseline[pid] = []
                baseline_opts[pid] = []
            baseline[pid].append(qa_info[t]["AnswerValue"])
            baseline_opts[pid].append(ord(qa_info[t]["Answer"]) - ord("A"))
    if not os.path.exists(os.path.join(args.save_root, "stats/ans_distr/sacc/train/")):
        os.makedirs(os.path.join(args.save_root, "stats/ans_distr/sacc/train/"))
        os.makedirs(os.path.join(args.save_root, "stats/ans_distr/sacc/val"))
        os.makedirs(os.path.join(args.save_root, "stats/ans_distr/sacc/test"))
        os.makedirs(os.path.join(args.save_root, "stats/ans_distr/oacc/train"))
        os.makedirs(os.path.join(args.save_root, "stats/ans_distr/oacc/val"))
        os.makedirs(os.path.join(args.save_root, "stats/ans_distr/oacc/test"))
    tot_baseline_sacc_greedy = 0.0
    tot_baseline_sacc_bestK = 0.0
    tot_rand_sacc = 0.0
    tot_baseline_oacc_greedy = 0.0
    tot_baseline_oacc_bestK = 0.0
    tot_rand_oacc = 0.0
    baseline_sacc = {}
    baseline_oacc = {}
    baseline_rand_sacc = {}
    overall_baseline_sacc = {}
    overall_baseline_oacc = {}
    overall_baseline_rand = {}
    for key in baseline.keys():
        x = plot_baseline_perf(args, baseline[key], key, tot, split, "sacc")
        baseline_sacc[key] = (x.argmax(), x.max(), topK(x), len(x))  # the class and the value.
        x = plot_baseline_perf(args, baseline_opts[key], key, tot, split, "oacc")
        baseline_oacc[key] = (x.argmax(), x.max(), topK(x), len(x))  # the class and the value.
        baseline_rand_sacc[key] = 1 / gv.NUM_CLASSES_PER_PUZZLE[key]

        tot_baseline_sacc_greedy += baseline_sacc[key][1]
        tot_baseline_sacc_bestK += baseline_sacc[key][2]
        tot_rand_sacc += 1 / gv.NUM_CLASSES_PER_PUZZLE[key]

        tot_baseline_oacc_greedy += baseline_oacc[key][1]
        tot_baseline_oacc_bestK += baseline_oacc[key][2]
        tot_rand_oacc += 1 / 5.0

        if True:  # log:
            print(
                "baseline %s class = %d freq = %f bestK_acc=%f percent num_classes=%d"
                % (key, baseline_sacc[key][0], baseline_sacc[key][1], baseline_sacc[key][2], baseline_sacc[key][3])
            )
        overall_baseline_sacc[key] = baseline_sacc[key][1]
        overall_baseline_oacc[key] = baseline_oacc[key][1]
        overall_baseline_rand[key] = baseline_rand_sacc[key]
    print("\n\n")
    tot_keys = len(baseline.keys())
    print(
        "overall baseline (%d puzzles) Greedy: top-1 sacc/oacc = %0.4f/%0.4f Greedy: top-K sacc/oacc=%0.4f/%0.4f random sacc=%0.4f "
        % (
            len(baseline.keys()),
            tot_baseline_sacc_greedy / tot_keys,
            tot_baseline_oacc_greedy / tot_keys,
            tot_baseline_sacc_bestK / tot_keys,
            tot_baseline_oacc_bestK / tot_keys,
            tot_rand_sacc / tot_keys,
        )
    )

    base_sacc_list = np.zeros(
        gv.num_puzzles + 1,
    )
    base_oacc_list = np.zeros(
        gv.num_puzzles + 1,
    )
    rand_sacc_list = np.zeros(
        gv.num_puzzles + 1,
    )
    for key in baseline.keys():
        base_sacc_list[int(key)] = overall_baseline_sacc[key]
        base_oacc_list[int(key)] = overall_baseline_oacc[key]
        rand_sacc_list[int(key)] = overall_baseline_rand[key]

    # print category-wise performances. # copied from  utils.
    puzzles = utils.read_dataset_info(gv.VILPS_DATASET_INFO_FILE)
    cls_mean = lambda x, idx: np.array([x[int(ii)] for ii in idx]).mean()
    get_int_set = lambda x: set([int(ii) for ii in x])
    class_avg_perf = {}
    classes = ["counting", "math", "logic", "path", "algebra", "measure", "spatial", "pattern"]
    print(classes)
    print("Greedy %s" % (split))
    for kk in classes:
        idx_list = np.array(list(get_int_set(puzzles[kk]).intersection(get_int_set(baseline.keys()))))
        class_avg_perf[kk] = (
            cls_mean(base_sacc_list, idx_list),
            cls_mean(base_oacc_list, idx_list),
            cls_mean(rand_sacc_list, idx_list),
        )
        print("%0.3f/%0.3f & " % (class_avg_perf[kk][0], class_avg_perf[kk][1]), end=" ")
    print("\nUniform %s" % (split))
    for kk in classes:
        print("%0.3f/%0.3f & " % (class_avg_perf[kk][2], 0.2), end=" ")
    print("\n\n")

    plt.figure(figsize=(30, 4))
    ax = plt.gca()
    ax.bar(np.arange(1, gv.num_puzzles + 1), 100.0 * base_sacc_list[1:])
    ax.set_xticks(np.arange(1, gv.num_puzzles + 1))  # , [str(i) for i in np.arange(1,num_puzzles+1)])
    plt.savefig(
        "%s/stats/ans_distr/%s/%s/base_sacc_perf_with_greedy_choices_%d.png" % (args.save_root, "sacc", split, tot)
    )
    plt.close()

    plt.figure(figsize=(30, 4))
    ax = plt.gca()
    ax.bar(np.arange(1, gv.num_puzzles + 1), 100.0 * base_oacc_list[1:])
    ax.set_xticks(np.arange(1, gv.num_puzzles + 1))  # , [str(i) for i in np.arange(1,num_puzzles+1)])
    plt.savefig(
        "%s/stats/ans_distr/%s/%s/base_oacc_perf_with_greedy_choices_%d.png" % (args.save_root, "oacc", split, tot)
    )
    plt.close()

    np.save("%s/stats/baseline_%d_%s.npy" % (args.save_root, tot, split), [baseline_sacc, baseline_oacc])
    return baseline_sacc, baseline_oacc
