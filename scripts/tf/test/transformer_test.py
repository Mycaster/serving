# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from grpc.beta import implementations
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import re
import time
import csv
import json
import sys
import numpy as np
import multiprocessing
import argparse


class TransformerClient(object):
    def __init__(self):
        channel = implementations.insecure_channel('127.0.0.1', 8500)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel._channel)

    def predict(self, request):
        try:
            start_time = time.time()
            self.stub.Predict(request, 10.0)
            ttl = (time.time() - start_time) * 1000
            print('success:{:.0f}ms'.format(ttl))
            return ttl
        except Exception as e:  # deadline exception
            print('failed with:', e)
            return None


def batch_test(id, query_per_process, batch):
    test_file = "test_data.csv"
    with open("word_dict.txt", 'r', encoding='U8') as f:
        words = [word.strip() for word in f.readlines()]

    idx2word = {idx: word for idx, word in enumerate(words)}
    word2idx = {word: idx for idx, word in idx2word.items()}
    client = TransformerClient()

    # 句子转化为 id
    def seq2ids(seq):
        seq = re.sub(r'\s+', '', seq).strip()  # 去除所有空格
        ids = []
        for c in seq:
            if c in word2idx:  # oov词忽略
                ids.append(word2idx[c])
        ids.append(1)  # 增加<EOS>
        return ids

    # input_rows: 原始数据的多行输入
    def make_batch_request(input_rows):
        seq2idx_list = []
        for row in input_rows:
            seq2idx_list.append(seq2ids(row))  # 取聊天内容

        inputs = []
        for seq in seq2idx_list:
            input = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "inputs": tf.train.Feature(int64_list=tf.train.Int64List(value=seq))
                    }
                )
            )
            inputs.append(input.SerializeToString())

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'transformer'
        request.inputs["input"].CopyFrom(tf.contrib.util.make_tensor_proto(
            # input, shape=[1]))
            inputs, shape=[len(inputs)]))

        return request

    ttls = []
    failed_querys = 0
    total_querys = 0
    with open(test_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        rows = []
        for row in csv_reader:
            rows.append(row[1])

        # 按 batch
        for i in range(0, len(rows), batch):
            total_querys += 1
            request = make_batch_request(rows[i:i + batch])
            ttl = client.predict(request)
            if ttl is None:
                failed_querys += 1
            else:
                ttls.append(ttl)

            if query_per_process > 0 and total_querys >= query_per_process:
                break

    success_rate = round(float(total_querys - failed_querys) / float(total_querys), 2)
    ttls = sorted(ttls)
    min_ttl = round(ttls[0], 2) if len(ttls) > 0 else 0.00
    max_ttl = round(ttls[-1], 2) if len(ttls) > 0 else 0.00
    aver_ttl = np.round(np.mean(ttls), 2) if len(ttls) > 0 else 0.00
    result = {
        "process_id": id,
        "total_querys": total_querys,
        "failed_querys": failed_querys,
        "success_rate": success_rate,
        "min_ttl": min_ttl,
        "max_ttl": max_ttl,
        "aver_ttl": aver_ttl,
        "aver_ttl_per_sample": round(aver_ttl / batch, 2),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--num_process', type=int, default=3)
    parser.add_argument('-n', '--query_per_process', type=int, default=500)
    parser.add_argument('-b', '--batch', type=int, default=1)
    args = parser.parse_args()
    num_process = args.num_process
    query_per_process = args.query_per_process
    batch = args.batch

    start_time = time.time()
    pool = multiprocessing.Pool(processes=num_process)
    result = []
    for i in range(num_process):
        result.append(pool.apply_async(batch_test, (i, query_per_process, batch)))

    pool.close()
    pool.join()
    interval = (time.time() - start_time)
    qps = (query_per_process * num_process) /(batch* interval)
    print("num_process=%d, querys_per_process=%d, batch=%d, qps=%.2f" %
          (num_process, query_per_process, batch, qps))
    for res in result:
        print(res.get())


if __name__ == "__main__":
    main()
