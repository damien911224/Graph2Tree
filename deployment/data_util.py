import random
import json
import copy
import re
import numpy as np
import codecs


def read_json_mac(path):
    # this is for mac
    with codecs.open(path, 'r', encoding="utf-8-sig") as f:
        file = json.load(f)
    return file

def read_json(path):
    with open(path, 'r') as f:
        file = json.load(f)
    return file


def load_data(path):
    questions = read_json_mac(path)
    return questions


def transfer_num(data):
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
