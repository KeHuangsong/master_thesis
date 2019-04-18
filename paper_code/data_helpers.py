import numpy as np
import re
from sklearn.preprocessing import LabelBinarizer


class DataHelper(object):
    def split_str(self, string):
        words = string.strip().split(" ")
        return [w for w in words if w]

    def load_data_and_labels(self, data_file):
        """
        Loads title data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        x_text = list(open(data_file, "r").readlines())
        cates = [s.split('\t')[0] for s in x_text if len(s.strip().split('\t')) == 2]
        y = [list(v) for v in LabelBinarizer().fit_transform(cates)]
        x_text = [self.split_str(s.strip().split('\t')[1]) for s in x_text if len(s.strip().split('\t')) == 2]
        return [x_text, y]

    def batch_iter(self, x, y, Len, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        x = np.array(x)
        y = np.array(y)
        Len = np.array(Len)
        data_size = len(x)
        num_batches_per_epoch = int((data_size-1)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_x = x[shuffle_indices]
                shuffled_y = y[shuffle_indices]
                shuffled_len = Len[shuffle_indices]
            else:
                shuffled_x = x
                shuffled_y = y
                shuffled_len = Len
            for batch_num in range(num_batches_per_epoch-1):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                rx = []
                for v in shuffled_x[start_index:end_index]:
                    rx += v
                yield np.array(rx).reshape([-1, 1]),\
                      np.array(shuffled_y[start_index:end_index]),\
                      np.array(shuffled_len[start_index:end_index])
