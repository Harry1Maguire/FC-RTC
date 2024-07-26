import time
from pcap_proc4 import Np, Nb
from model import *
from sklearn.metrics import f1_score, confusion_matrix
import random

BATCH = 16


def stream_to_data(streams: dict, token: int):
    HDRs = []
    PAYs = []
    tokens = []
    for stream_hash in streams:
        stream = streams[stream_hash]
        HDR = []
        PAY = []
        for hdr in stream.HDR:
            HDR.extend(hdr)
        for pay in stream.PAY:
            PAY.extend(pay)
        for i in range(len(HDR), Np * 4):
            HDR.append(0.)
        for i in range(len(PAY), Nb):
            PAY.append(0.)
        HDR = np.array(HDR, dtype=np.float32)
        PAY = np.array(PAY, dtype=np.float32)
        HDRs.append(HDR)
        PAYs.append(PAY)
        tokens.append(token)
    # shuffle
    # print(sum(tokens))
    HDRs = np.array(HDRs, dtype=np.float32)
    PAYs = np.array(PAYs, dtype=np.float32)
    tokens = np.array(tokens, dtype=np.float32)
    index = list(range(len(HDRs)))
    np.random.shuffle(index)
    HDRs = HDRs[index]
    PAYs = PAYs[index]
    tokens = tokens[index]
    div1 = int(len(HDRs) * 0.75)
    div2 = int(len(HDRs) * 0.85)
    train_set = myPreDataSet(HDRs[:div1], PAYs[:div1], tokens[:div1])
    val_set = myPreDataSet(HDRs[div1:div2], PAYs[div1:div2], tokens[div1:div2])
    test_set = myPreDataSet(HDRs[div2:], PAYs[div2:], tokens[div2:])
    return train_set, val_set, test_set


def creat_hash_for_datasplit(streams: dict):
    hash_list = []
    for stream_hash in streams:
        hash_list.append(stream_hash)
    n1 = int(len(hash_list) * 0.75)  # 训练集
    n2 = int(len(hash_list) * 0.10)  # 验证集
    n3 = int(len(hash_list) * 0.15)  # 测试集
    # 随机选取元素
    list1 = random.sample(hash_list, n1)
    list2 = random.sample([x for x in hash_list if x not in list1], n2)
    list3 = random.sample([x for x in hash_list if x not in list1 + list2], n3)

    return list1, list2, list3

def stream_to_data_hashlist(streams: dict, token: int, train_list, val_list, test_list):
    HDRs = []
    PAYs = []
    tokens = []
    for stream_hash in train_list:
        if stream_hash in streams:
            HDR = []
            PAY = []
            stream = streams[stream_hash]
            for hdr in stream.HDR:
                HDR.extend(hdr)
            for pay in stream.PAY:
                PAY.extend(pay)
            for i in range(len(HDR), Np * 4):
                HDR.append(0.)
            for i in range(len(PAY), Nb):
                PAY.append(0.)
            HDR = np.array(HDR, dtype=np.float32)
            PAY = np.array(PAY, dtype=np.float32)
            HDRs.append(HDR)
            PAYs.append(PAY)
            tokens.append(token)
    for stream_hash in val_list:
        if stream_hash in streams:
            HDR = []
            PAY = []
            stream = streams[stream_hash]
            for hdr in stream.HDR:
                HDR.extend(hdr)
            for pay in stream.PAY:
                PAY.extend(pay)
            for i in range(len(HDR), Np * 4):
                HDR.append(0.)
            for i in range(len(PAY), Nb):
                PAY.append(0.)
            HDR = np.array(HDR, dtype=np.float32)
            PAY = np.array(PAY, dtype=np.float32)
            HDRs.append(HDR)
            PAYs.append(PAY)
            tokens.append(token)
    for stream_hash in test_list:
        if stream_hash in streams:
            HDR = []
            PAY = []
            stream = streams[stream_hash]
            for hdr in stream.HDR:
                HDR.extend(hdr)
            for pay in stream.PAY:
                PAY.extend(pay)
            for i in range(len(HDR), Np * 4):
                HDR.append(0.)
            for i in range(len(PAY), Nb):
                PAY.append(0.)
            HDR = np.array(HDR, dtype=np.float32)
            PAY = np.array(PAY, dtype=np.float32)
            HDRs.append(HDR)
            PAYs.append(PAY)
            tokens.append(token)
    HDRs = np.array(HDRs, dtype=np.float32)
    PAYs = np.array(PAYs, dtype=np.float32)
    tokens = np.array(tokens, dtype=np.float32)
    print(len(HDRs))
    div1 = int(len(HDRs) * 0.75)
    div2 = int(len(HDRs) * 0.85)
    train_set = myPreDataSet(HDRs[:div1], PAYs[:div1], tokens[:div1])
    val_set = myPreDataSet(HDRs[div1:div2], PAYs[div1:div2], tokens[div1:div2])
    test_set = myPreDataSet(HDRs[div2:], PAYs[div2:], tokens[div2:])
    return train_set, val_set, test_set

def stream_to_data_unbalanced_type(streams: dict, token: int, train_list, val_list, test_list):
    HDRs = []
    PAYs = []
    tokens = []
    for stream_hash in train_list:
        if stream_hash in streams:
            HDR = []
            PAY = []
            stream = streams[stream_hash]
            for hdr in stream.HDR:
                HDR.extend(hdr)
            for pay in stream.PAY:
                PAY.extend(pay)
            for i in range(len(HDR), Np * 4):
                HDR.append(0.)
            for i in range(len(PAY), Nb):
                PAY.append(0.)
            HDR = np.array(HDR, dtype=np.float32)
            PAY = np.array(PAY, dtype=np.float32)
            HDRs.append(HDR)
            PAYs.append(PAY)
            tokens.append(token)
    for stream_hash in val_list:
        if stream_hash in streams:
            HDR = []
            PAY = []
            stream = streams[stream_hash]
            for hdr in stream.HDR:
                HDR.extend(hdr)
            for pay in stream.PAY:
                PAY.extend(pay)
            for i in range(len(HDR), Np * 4):
                HDR.append(0.)
            for i in range(len(PAY), Nb):
                PAY.append(0.)
            HDR = np.array(HDR, dtype=np.float32)
            PAY = np.array(PAY, dtype=np.float32)
            HDRs.append(HDR)
            PAYs.append(PAY)
            tokens.append(token)
    for stream_hash in test_list:
        if stream_hash in streams:
            HDR = []
            PAY = []
            stream = streams[stream_hash]
            for hdr in stream.HDR:
                HDR.extend(hdr)
            for pay in stream.PAY:
                PAY.extend(pay)
            for i in range(len(HDR), Np * 4):
                HDR.append(0.)
            for i in range(len(PAY), Nb):
                PAY.append(0.)
            HDR = np.array(HDR, dtype=np.float32)
            PAY = np.array(PAY, dtype=np.float32)
            HDRs.append(HDR)
            PAYs.append(PAY)
            tokens.append(token)
    HDRs = np.array(HDRs, dtype=np.float32)
    PAYs = np.array(PAYs, dtype=np.float32)
    tokens = np.array(tokens, dtype=np.float32)
    tokens = tokens[:, None]
    return HDRs, PAYs, tokens

def stream_to_data_share(streams: dict, token: int, train_list, val_list, test_list, i):
    HDRs = []
    PAYs = []
    tokens = []
    for stream_hash in train_list:
        if stream_hash in streams:
            HDR = []
            PAY = []
            stream = streams[stream_hash]
            for hdr in stream.HDR:
                HDR.extend(hdr)
            for pay in stream.PAY:
                PAY.extend(pay)
            for i in range(len(HDR), Np * 4):
                HDR.append(0.)
            for i in range(len(PAY), Nb):
                PAY.append(0.)
            HDR = np.array(HDR, dtype=np.float32)
            PAY = np.array(PAY, dtype=np.float32)
            HDRs.append(HDR)
            PAYs.append(PAY)
            tokens.append(token)
    for stream_hash in val_list:
        if stream_hash in streams:
            HDR = []
            PAY = []
            stream = streams[stream_hash]
            for hdr in stream.HDR:
                HDR.extend(hdr)
            for pay in stream.PAY:
                PAY.extend(pay)
            for i in range(len(HDR), Np * 4):
                HDR.append(0.)
            for i in range(len(PAY), Nb):
                PAY.append(0.)
            HDR = np.array(HDR, dtype=np.float32)
            PAY = np.array(PAY, dtype=np.float32)
            HDRs.append(HDR)
            PAYs.append(PAY)
            tokens.append(token)
    for stream_hash in test_list:
        if stream_hash in streams:
            HDR = []
            PAY = []
            stream = streams[stream_hash]
            for hdr in stream.HDR:
                HDR.extend(hdr)
            for pay in stream.PAY:
                PAY.extend(pay)
            for i in range(len(HDR), Np * 4):
                HDR.append(0.)
            for i in range(len(PAY), Nb):
                PAY.append(0.)
            HDR = np.array(HDR, dtype=np.float32)
            PAY = np.array(PAY, dtype=np.float32)
            HDRs.append(HDR)
            PAYs.append(PAY)
            tokens.append(token)
    HDRs = np.array(HDRs, dtype=np.float32)
    PAYs = np.array(PAYs, dtype=np.float32)
    tokens = np.array(tokens, dtype=np.float32)
    div1 = int(len(HDRs) * 0.75)
    div2 = int(len(HDRs) * 0.85)
    a1 = int(len(HDRs) * 0.75 * i)
    a2 = int(len(HDRs) * 0.75 * (i+1))
    train_set = myPreDataSet(HDRs[a1:a2], PAYs[a1:a2], tokens[a1:a2])
    val_set = myPreDataSet(HDRs[div1:div2], PAYs[div1:div2], tokens[div1:div2])
    test_set = myPreDataSet(HDRs[div2:], PAYs[div2:], tokens[div2:])
    return train_set, val_set, test_set

