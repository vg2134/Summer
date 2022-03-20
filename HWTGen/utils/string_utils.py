import numpy as np
import sys
import pandas as pd

import torch


def str2label_single(value, characterToIndex={}, unknown_index=None):
    if unknown_index is None:
        unknown_index = len(characterToIndex)

    label = []
    for v in value:
        if v not in characterToIndex:
            continue
            # raise "Unknown Charactor to Label conversion"
        label.append(characterToIndex[v])
    return np.array(label, np.uint32)


def label2input_single(value, num_of_inputs, char_break_interval):
    idx1 = len(value) * (char_break_interval + 1) + char_break_interval
    idx2 = num_of_inputs + 1
    input_data = [[0 for i in range(idx2)] for j in range(idx1)]

    cnt = 0
    for i in range(char_break_interval):
        input_data[cnt][idx2 - 1] = 1
        cnt += 1

    for i in range(len(value)):
        if value[i] == 0:
            input_data[cnt][idx2 - 1] = 1
        else:
            input_data[cnt][value[i] - 1] = 1
        cnt += 1

        for i in range(char_break_interval):
            input_data[cnt][idx2 - 1] = 1
            cnt += 1

    return np.array(input_data)


def label2str_single(label, indexToCharacter, asRaw, spaceChar="~"):
    string = u""
    for i in range(len(label)):
        if label[i] == 0:
            if asRaw:
                string += spaceChar
            else:
                break
        else:
            val = label[i]
            string += indexToCharacter[val]
    return string


def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
    softmax = x_exp / x_exp_row_sum
    return softmax


def find_sub_max(arr, n):
    try:
        arr_ = arr
        for i in range(n - 1):
            arr_ = arr
            arr_[np.argmax(arr_, axis=-1)] = np.min(arr)
            arr = arr_
        return np.argmax(arr_, axis=-1)
    except:
        return np.argmax(arr, -1)


def naive_decode(output):
    # rawPredData = np.argmax(output, axis=1)
    predData_ = list()
    rawPredData_ = list()
    print(output.shape)
    for j in range(1, 3):
        rawPredData = find_sub_max(output, j)
        predData = []
        for i in range(len(output)):
            if rawPredData[i] != 0 and not (i > 0 and rawPredData[i] == rawPredData[i - 1]):
                predData.append(rawPredData[i])
        print(len(predData))
        print(len(rawPredData))
        # raise NotImplementedError
        predData_.append(predData)
        rawPredData_.append(rawPredData)
    return predData_, list(rawPredData_)
