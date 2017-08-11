# import re
# import numpy as np
import os
import sys

# reload(sys)
sys.setdefaultencoding('utf8')


def load_files(log_directory, is_spam):
    txts = []

    for f in os.listdir(log_directory):
        with open(log_directory + f, 'rb') as infile:
            subject = infile.readline()
            body = infile.read()
            subject = subject.decode('utf-8', 'ignore').encode("utf-8")
            body = body.decode('utf-8', 'ignore').encode("utf-8")
            txts.append((subject, infile.read(), is_spam))

    return txts


if __name__ == '__main__':
    txts = load_files('data/enron1/spam/')
