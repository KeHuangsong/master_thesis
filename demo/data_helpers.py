import numpy as np
import re


class DataHelper(object):
    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = re.sub("\'", " ' ",string)
        return string.strip().lower()

    def load_data_and_labels(self, positive_data_file, negative_data_file):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        # Load data from files
        positive_examples = list(open(positive_data_file, "r").readlines())
        positive_examples = [s.strip() for s in positive_examples]
        negative_examples = list(open(negative_data_file, "r").readlines())
        negative_examples = [s.strip() for s in negative_examples]
        # Split by words
        x_text = positive_examples + negative_examples
        x_text = [self.clean_str(sent) for sent in x_text]
        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        y = np.concatenate([positive_labels, negative_labels], 0)
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
