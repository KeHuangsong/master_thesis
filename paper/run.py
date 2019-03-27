# -*- coding: utf-8 -*-

import os
import json
import argparse
from multiprocessing import Process
from train_pre_trained import ModelTrain

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('-model', type=str, default=json.dumps(['origin', 'weighted_origin']))
parser.add_argument('-dropout_keep_prob', type=float, default=0.5)
parser.add_argument('-l2_reg_lambda', type=float, default=0)
parser.add_argument('-batch_size', type=int, default=100)
parser.add_argument('-refresh_data', type=str, default='')
args = parser.parse_args()


data_file = "./data/raw_title_data.txt"
dev_sample_percentage = 0.1
resize_n = 100
dropout_keep_prob = args.dropout_keep_prob
l2_reg_lambda = args.l2_reg_lambda
batch_size = args.batch_size
num_epochs = 200
evaluate_every = 100
checkpoint_every = 100
num_checkpoints = 5
allow_soft_placement = True
log_device_placement = False
model_train = ModelTrain(dev_sample_percentage, data_file, dropout_keep_prob, batch_size, num_epochs,
                         evaluate_every, checkpoint_every, num_checkpoints, allow_soft_placement,
                         log_device_placement, resize_n, l2_reg_lambda)


if __name__ == "__main__":
    if args.refresh_data:
        os.system('rm -r ./data')
    if not os.path.exists('./data'):
        os.system('mkdir ./data')
    print args
    model_train.get_data('run.log')
    models = json.loads(args.model)
    ps = []
    if 'origin' in models:
        p = Process(target=model_train.start, args=('origin', './data/origin.out'))
        ps.append(p)
    if 'weighted_origin' in models:
        p = Process(target=model_train.start, args=('weighted_origin', './data/weighted_origin.out'))
        ps.append(p)
    if 'origin_pro' in models:
        p = Process(target=model_train.start, args=('origin_pro', './data/origin_pro.out'))
        ps.append(p)
    if 'resized' in models:
        p = Process(target=model_train.start, args=('resized', './data/resized.out'))
        ps.append(p)
    if 'both' in models:
        p = Process(target=model_train.start, args=('both', './data/both.out'))
        ps.append(p)
    if 'ssp' in models:
        p = Process(target=model_train.start, args=('ssp', './data/ssp.out'))
        ps.append(p)
    [p.start() for p in ps]
    [p.join() for p in ps]



