# -*- coding: utf-8 -*-

import os

if not os.path.exists('./data'):
    os.system('mkdir ./data')
    os.system('python get_text_data.py')
os.system('./fastText-0.1.0/fasttext skipgram -input ./data/title_data.txt -output ./data/pre_trained_model')
os.system('hdfs dfs -rm /user/kehuangsong/pre_trained_vec/paper_pre_trained_model.bin')
os.system('hdfs dfs -rm /user/kehuangsong/pre_trained_vec/paper_pre_trained_model.vec')
os.system('hdfs dfs -copyFromLocal -f ./data/pre_trained_model.bin /user/kehuangsong/pre_trained_vec/paper_pre_trained_model.bin')
os.system('hdfs dfs -copyFromLocal -f ./data/pre_trained_model.vec /user/kehuangsong/pre_trained_vec/paper_pre_trained_model.vec')

