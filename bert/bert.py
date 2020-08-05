from bert_serving.client import BertClient
import socket
import pickle
import sys
import numpy

numpy.set_printoptions(threshold=sys.maxsize)

def get_embd(sentences):
    bc = BertClient()
    return bc.encode(sentences,show_tokens=True)


def get_word_econded():
    with open("word_dict.txt", "rb") as fp:
        return pickle.load(fp)
