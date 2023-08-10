#!/usr/bin/env python3
# Copyright (c) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT
#
import json
import os
import os.path as osp
import pdb
import pickle as pkl
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import globvars as gv


def fix_acc(acc_list):
    """removes accuracy for puzzles in gv.puzzles_not_included"""
    idx = np.array(list(set(np.arange(1, gv.num_puzzles + 1)).difference(set(gv.puzzles_not_included))))
    new_acc_list = acc_list[idx - 1]
    return new_acc_list


def get_icon_dataset_classes(icon_path):
    """returns the classes in ICONs-50 dataset"""
    with open(icon_path, "r") as f:
        icon_classes = f.readlines()
    return [ii.rstrip() for ii in icon_classes]


def print_puzz_acc(args, puzz_acc, log=True):
    to_int = lambda x: np.array(list(x)).astype("int")
    cls_mean = lambda x, idx, pids: np.array([x[int(ii)] for ii in idx]).sum() / len(
        set(to_int(idx)).intersection(set(to_int(pids)))
    )
    acc_list = np.zeros(
        gv.num_puzzles + 1,
    )
    opt_acc_list = np.zeros(
        gv.num_puzzles + 1,
    )

    if not os.path.exists(os.path.join(args.save_root, "results/%d/" % (gv.seed))):
        os.makedirs(os.path.join(args.save_root, "results/%d/" % (gv.seed)))

    if len(puzz_acc.keys()) > 10:
        for k, key in enumerate(puzz_acc.keys()):
            acc = 100.0 * puzz_acc[key][0] / puzz_acc[key][2]
            oacc = 100.0 * puzz_acc[key][1] / puzz_acc[key][2]
            acc_list[int(key)] = acc
            opt_acc_list[int(key)] = oacc
        if log:
            for t in range(1, gv.num_puzzles + 1):
                print("%d opt_acc=%0.2f acc=%0.2f" % (t, opt_acc_list[t], acc_list[t]), end="\t")
                if t % 5 == 0:
                    print("\n")
            print("\n\n")

            puzzles = read_dataset_info(gv.SMART_DATASET_INFO_FILE)
            class_avg_perf = {}
            classes = ["counting", "math", "logic", "path", "algebra", "measure", "spatial", "pattern"]
            print(classes)
            for kk in classes:
                idx_list = puzzles[kk]
                class_avg_perf[kk] = (
                    cls_mean(acc_list, idx_list, list(puzz_acc.keys())),
                    cls_mean(opt_acc_list, idx_list, list(puzz_acc.keys())),
                )
                print("%0.1f/%0.1f & " % (class_avg_perf[kk][0], class_avg_perf[kk][1]), end=" ")
            print("\n\n")

        fig = plt.figure(figsize=(30, 4))
        ax = plt.gca()
        ax.bar(np.arange(1, gv.num_actual_puzz), fix_acc(acc_list[1:]))
        ax.set_xticks(np.arange(1, gv.num_actual_puzz))
        ax.set_xlabel("puzzle ids", fontsize=16)
        ax.set_ylabel("$O_{acc}$ %", fontsize=20)
        fig.tight_layout()
        plt.savefig(os.path.join(args.save_root, "results/%d/acc_perf_scores_1.png" % (gv.seed)))
        plt.close()

        fig = plt.figure(figsize=(30, 4))
        ax = plt.gca()
        ax.bar(np.arange(1, gv.num_actual_puzz), fix_acc(opt_acc_list[1:]))
        ax.set_xticks(np.arange(1, gv.num_actual_puzz))  # , [str(i) for i in np.arange(1,num_puzzles+1)])
        ax.set_xlabel("puzzle ids", fontsize=16)
        ax.set_ylabel("$S_{acc}$ %", fontsize=20)
        fig.tight_layout()
        plt.savefig(os.path.join(args.save_root, "results/%d/opt_acc_perf_scores_1.png" % (gv.seed)))
        plt.close()
    else:
        for key in puzz_acc.keys():
            acc = 100.0 * puzz_acc[key][0] / puzz_acc[key][2]
            opt_acc = 100.0 * puzz_acc[key][1] / puzz_acc[key][2]
            if log:
                print("%s opt_acc=%0.2f acc=%0.2f" % (key, opt_acc, acc))
            acc_list[int(key)] = acc
            opt_acc_list[int(key)] = opt_acc

        plt.figure()
        plt.bar(np.arange(gv.num_puzzles + 1), acc_list)
        plt.savefig(os.path.join(args.save_root, "results/%d/acc_perf_scores.png" % (gv.seed)))
        plt.close()
        plt.figure()
        plt.bar(np.arange(gv.num_puzzles + 1), opt_acc_list)
        plt.savefig(os.path.join(args.save_root, "results/%d/opt_acc_perf_scores.png" % (gv.seed)))
        plt.close()


def get_option_sel_acc(pred_ans, opts, answer, answer_values, pid):
    """converts a predicted answer to one of the given multiple choice options.
    opts is b x num_options matrix"""

    def get_op_str(ii):
        return gv.signs[int(str(ii)[0]) - 1] + str(ii)[1:] if ii >= 10 else gv.signs[0] + str(ii)

    if pid in gv.SEQ_PUZZLES:
        result = np.abs(answer_values - pred_ans).sum(axis=1) == 0
    elif pid in [32, 69, 82, 84, 95, 98, 51, 66, 44, 68]:
        result = [pred_ans[i] == answer[i] for i in range(len(pred_ans))]
    else:
        try:
            result = (
                np.abs(opts.astype("float") - pred_ans.unsqueeze(1).cpu().numpy()).argmin(axis=1)
                == answer.cpu().numpy()
            )
        except:
            result = [pred_ans[i] == answer[i] for i in range(len(pred_ans))]
            print("error!!")
            pdb.set_trace()
    return np.array(result)


def read_dataset_info(csvfilename):
    import csv

    qa_info = {}
    with open(csvfilename, newline="") as csvfile:
        datareader = csv.DictReader(csvfile)
        for row in datareader:
            key = str(row["type"]).lower()
            if key not in qa_info.keys():
                qa_info[key] = [row["id"]]
            else:
                qa_info[key].append(row["id"])
    assert np.array([len(qa_info[key]) for key in qa_info.keys()]).sum() == 101
    return qa_info


def read_csv(csvfilename, puzzle_id):
    import csv

    qa_info = []
    with open(csvfilename, newline="") as csvfile:
        datareader = csv.DictReader(csvfile)
        for row in datareader:
            row["puzzle_id"] = str(puzzle_id)
            if len(row["A"]) == 0:
                row["A"] = "A"
                row["B"] = "B"
                row["C"] = "C"
                row["D"] = "D"
                row["E"] = "E"
            qa_info.append(row)
    return qa_info


def pad_with_max_val(gt_list, val):
    """if the number of elements in gt is less than MAX_DECODE_STEPS, we pad it with the max value in a class"""
    if len(gt_list) < gv.MAX_DECODE_STEPS:
        gt_list = (
            gt_list
            + (
                np.ones(
                    gv.MAX_DECODE_STEPS - len(gt_list),
                )
                * val
            ).tolist()
        )
    return gt_list


def str_replace(ans):
    ans = ans.replace(" hours", "")
    ans = ans.replace(" hour", "").replace(" cm", "")
    ans = ans.replace(" km", "")
    return ans


def str_replace_(info, ans_opt):
    ans = info[ans_opt]
    ans = ans.replace(" hours", "")
    ans = ans.replace(" hour", "").replace(" cm", "")
    ans = ans.replace(" km", "")
    ans = ans.replace("Impossible", "0")
    info[ans_opt] = ans
    return ans


def get_val(qinfo, ans_opt, is_one_of_option=False):
    """get the value of the answer option. This code also encodes the value into a number by removing extreneous strings"""
    """ is_one_of_option is True, when ans_opt is one of the options, need not be the correct answer option."""
    where = lambda x, y: np.where(np.array(x) == y)[0][0]

    pid = int(qinfo["puzzle_id"])
    if pid in gv.SEQ_PUZZLES:
        ans = qinfo[ans_opt]
        if pid == 16:
            ans_opt_val = [int(ii) for ii in ans.replace("and", ",").replace(", ,", ",").replace(" ", "").split(",")]
            ans_opt_val = pad_with_max_val(ans_opt_val, 26)
        elif pid == 18:
            ans_opt_val = [int(ii) for ii in ans.split("-")]
            ans_opt_val = pad_with_max_val(ans_opt_val, 5)
        elif pid == 35:
            ans_opt_val = [
                ord(ii) - ord("A") for ii in ans.replace("and", ",").replace(", ,", ",").replace(" ", "").split(",")
            ]
            ans_opt_val = pad_with_max_val(ans_opt_val, 5)
        elif pid == 39:
            ans_opt_val = [ord(ii) - ord("A") for ii in list(ans)]
            ans_opt_val = pad_with_max_val(ans_opt_val, 26)
        elif pid == 63:
            ans_opt_val = [
                int(ii)
                for ii in ans.replace("and", ",")
                .replace("or", ",")
                .replace(", ,", ",")
                .replace("only", "")
                .replace(" ", "")
                .split(",")
            ]
            key = str(63)
            if key in gv.NUM_CLASSES_PER_PUZZLE:
                ans_opt_val = pad_with_max_val(ans_opt_val, gv.NUM_CLASSES_PER_PUZZLE[key] - 1)
        elif pid == 100:
            ans_opt_val = [ord(ii) - ord("A") for ii in list(ans)]
            ans_opt_val = pad_with_max_val(ans_opt_val, 26)
        ans_opt_val = np.array(ans_opt_val)

    elif pid == 58:
        # puzzle 58 has answers as <operator><one digit number>, e.g./4,-5, etc.
        # we use +=1, -=2, x=3, /=4. so /4 will be 44, -5=25, +2= 2.
        ans_opt_val = qinfo[ans_opt]
        ans_opt_val = (where(gv.signs, ans_opt_val[0]) + 1) * 10 + int(ans_opt_val[1:])
    elif pid == 25:
        # we need to fix the time in AM/PM format properly.
        ans = qinfo[ans_opt]
        ans_opt_val = int(ans.replace(":00 AM", "").replace(":00 PM", ""))
        if ans.find("PM") > -1:
            ans_opt_val += 12
    else:
        try:
            ans_opt_val = int(qinfo[ans_opt])
        except:
            if len(qinfo[ans_opt]) > 0:
                try:
                    ans_opt_val = ord(qinfo[ans_opt]) - ord("A")
                except:
                    try:
                        ans_opt_val = str_replace(qinfo[ans_opt])
                        ans_opt_val = ans_opt_val.replace("Impossible", "0")  # puzzle 58.
                        if int(qinfo["puzzle_id"]) == 1:  # if the puzzle id is 1, then the options are icon classes.
                            ans_opt_val = "_".join(ans_opt_val.split(" "))
                            if ans_opt_val in gv.icon_class_ids:
                                ans_opt_val = where(gv.icon_class_ids, ans_opt_val)
                            elif ans_opt_val + "s" in gv.icon_class_ids:
                                ans_opt_val = where(gv.icon_class_ids, ans_opt_val + "s")
                        ans_opt_val = int(ans_opt_val)
                    except:
                        print(qinfo)
                        pdb.set_trace()
            else:
                ans_opt_val = ord(ans_opt) - ord("A")
    if not is_one_of_option:  # implies we are encoding the correct answer.
        qinfo["AnswerValue"] = ans_opt_val
    return ans_opt_val


def get_puzzle_class_info(args):
    #    global SEQ_PUZZLES, puzzle_diff_str, puzzle_diff
    puzzle_classes = {}
    for puzzle_id in args.puzzle_ids:
        puzzle_root = puzzle_id + "/" + gv.puzzle_diff_str[args.train_diff] + "/"
        csv_file = "puzzle_%s%s.csv" % (puzzle_id, gv.puzzle_diff[args.train_diff])
        qa_info = read_csv(os.path.join(args.data_root, puzzle_root, csv_file), puzzle_id)

        pid = int(puzzle_id)
        if pid not in gv.SEQ_PUZZLES:
            num_classes = np.array([get_val(qa, qa["Answer"]) for qa in qa_info]).max() + 1
        else:
            if pid in [16, 39, 100]:
                num_classes = 26 + 1  # if the output is a string of numbers, and the max classes is - max val.
            elif pid in [18, 35]:
                num_classes = 5 + 1  # the minus one is for end of items.
            elif pid in [63]:
                num_classes = np.array([get_val(qa, qa["Answer"]).max() for qa in qa_info]).max() + 1
        puzzle_classes[str(puzzle_id)] = num_classes
    return puzzle_classes


class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def set_gpu_devices(gpu_id):
    gpu = ""
    if gpu_id != -1:
        gpu = str(gpu_id)
    os.environ["CUDA_VOSIBLE_DEVICES"] = gpu


def load_file(filename):
    """
    load obj from filename
    :param filename:
    :return:
    """
    cont = None
    if not osp.exists(filename):
        print("{} not exist".format(filename))
        return cont
    if osp.splitext(filename)[-1] == ".csv":
        # return pd.read_csv(filename, delimiter= '\t', index_col=0)
        return pd.read_csv(filename, delimiter=",")
    with open(filename, "r") as fp:
        if osp.splitext(filename)[1] == ".txt":
            cont = fp.readlines()
            cont = [c.rstrip("\n") for c in cont]
        elif osp.splitext(filename)[1] == ".json":
            cont = json.load(fp)
    return cont


def save_file(obj, filename):
    """
    save obj to filename
    :param obj:
    :param filename:
    :return:
    """
    filepath = osp.dirname(filename)
    if filepath != "" and not osp.exists(filepath):
        os.makedirs(filepath)
    else:
        with open(filename, "w") as fp:
            json.dump(obj, fp, indent=4)


def pkload(file):
    data = None
    if osp.exists(file) and osp.getsize(file) > 0:
        with open(file, "rb") as fp:
            data = pkl.load(fp)
        # print('{} does not exist'.format(file))
    return data


def get_image(img):
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    img = img * 255
    img = img.cpu().numpy()
    img = img.astype("uint8")
    return Image.fromarray(img)


def pkdump(data, file):
    dirname = osp.dirname(file)
    if not osp.exists(dirname):
        os.makedirs(dirname)
    with open(file, "wb") as fp:
        pkl.dump(data, fp)


def get_puzzle_ids(args):
    puzzles = read_dataset_info(gv.SMART_DATASET_INFO_FILE)
    if args.puzzles == "all":
        puzzle_ids = os.listdir(args.data_root)
        puzzle_ids = np.array(puzzle_ids)[np.array([x.find(".") == -1 for x in puzzle_ids])]
        puzzle_ids = puzzle_ids.tolist()
        puzzle_ids_str = "all"
    elif args.puzzles in puzzles:
        puzzle_ids = puzzles[args.puzzles]
        puzzle_ids_str = args.puzzles
    else:
        puzzle_ids = args.puzzles.split(",")
        sorted_puzzle_ids = np.sort(np.array([int(ii) for ii in puzzle_ids]))
        puzzle_ids = [str(ii) for ii in sorted_puzzle_ids]
        puzzle_ids_str = "_".join(puzzle_ids)

    if args.monolithic:
        # remove sequential puzzles from the monolithic architecture.
        puzzle_ids = set(puzzle_ids).difference(set([str(ii) for ii in gv.SEQ_PUZZLES]))
        puzzle_ids = list(puzzle_ids)
        puzzle_ids_str = puzzle_ids_str + "_monolithic"

    return puzzle_ids_str, puzzle_ids


def backup_code_and_start_logger(args, log_path, seed):
    test = "test" if args.test else ""
    log_path = os.path.join(log_path, str(seed), test)
    if os.path.exists(log_path):
        log_path += "." + str(np.random.randint(0, high=100))
        print("test_path = %s" % (log_path))
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if not args.test:
        code_path = os.path.join(log_path, "code")
        if not os.path.exists(code_path):
            os.mkdir(code_path)
        print("saving code to %s" % (code_path))
        os.system("cp *.py %s" % (code_path))

    with open("%s/cmd_line.txt" % (log_path), "w") as cmd:
        cmd.write(str(sys.argv))

    log_file = os.path.join(log_path, "%d.log" % (seed))
    sys.stdout = Logger(log_file)
    print("logging results to %s" % (log_file))


#
