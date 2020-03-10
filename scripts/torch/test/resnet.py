# -*- coding: utf-8 -*-
"""
@author: husong
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf

import argparse
import torch 
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import time
from PIL import Image

def read_image_as_nparray(filename='Socks-clinton.jpg'):
    a = Image.open(filename) # 加载图片数据，返回的是一个PIL类型
    print(a.size)
    a.resize((224,224))
    print(a.size)
    b = np.array(a).transpose((2,0,1)) # 先将PIL类型转化成numpy类型，并且把数据变成浮点数
    return b

def read_images_from_torch():
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder("./", transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])))
    for i, (images, target) in enumerate(val_loader):
        print(images)
        return images.numpy()


class Client(object):
    def __init__(self, ip, port):
        channel = implementations.insecure_channel(ip, port)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel._channel)
        self.timeout = 1000

    def predict(self, input_image, model_name):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = model_name
        request.inputs["input"].CopyFrom(tf.contrib.util.make_tensor_proto(
            values=input_image))
            #shape=[1, 3, 224, 224]))
        try:
            start_time = time.time()
            response = self.stub.Predict(request, self.timeout)
            print('耗时:{:.0f}ms'.format((time.time() - start_time) * 1000))
        except Exception as e:  # deadline exception
            print(e)
            return None
        #print("request.inputs[\"input\"]", request.inputs["input"])
        return response.outputs["output"]  # (batch,beam,length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--ip', type=str, default='0.0.0.0')
    parser.add_argument('-port', '--port', type=int, default=8500)
    parser.add_argument('-m', '--model_name', type=str, default='resnet')

    args = parser.parse_args()
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
     #input_image = read_image_as_nparray()
    #input_image = np.array(tf.ones([1,3,224,224]))
    input_image = tf.random_normal(shape=[1,3,224,224], dtype=tf.float32).eval(session=sess)
    #input_image = tf.ones(shape=[1,3,224,224], dtype=tf.float32).eval(session=sess)
    

    #print("---input---:\n", input_image, "\n")
    print("input.shape:", input_image.shape)
    outputs = Client(args.ip, args.port).predict(input_image, args.model_name)
    #print("---output---:", outputs)


