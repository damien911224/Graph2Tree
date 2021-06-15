import json
import torch
import os
import codecs
from data_util import read_json_mac
from model_util import load_model
from src.pre_data import transfer_num

weight_path = ""
data_path = ""
problem_file = "problemsheet.json"
answer_file = "answersheet.json"



def load_data(path):
    questions = read_json_mac(path)
    return questions


if __name__ == "__main__":
    questions = load_data(problem_file)

    pairs, generate_nums, copy_nums = transfer_num(questions)
    print(pairs)