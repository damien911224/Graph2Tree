import json
import torch
import os
import codecs
from data_util import read_json_mac, extract, transfer_num_n_equation, prepare_infer_data, CustomDataSet
from model_util import load_model
from src.pre_data import transfer_num
from src.helper import index_batch_to_words, sentence_from_indexes

weight_path = ""
data_path = ""
problem_file = "problemsheet.json"
answer_file = "answersheet.json"

MAX_OUTPUT_LENGTH = 100

def evaluate_tree(input_batch, input_length, generate_nums, encoder, decoder, attention_decoder,
                  output_lang, num_pos, batch_graph, max_length=MAX_OUTPUT_LENGTH):
    pass


if __name__ == "__main__":

    data = extract(problem_file)

    pairs, copy_nums = transfer_num_n_equation(data)
    input_lang, test_pairs = prepare_infer_data(pairs, 5)
    # sent = index_batch_to_words([test_pairs[0]], [test_pairs[1]], input_lang)
    # sent = sentence_from_indexes(input_lang, test_pairs[0][0])
    for test_batch in test_pairs:
        sent = index_batch_to_words([test_batch[0]], [test_batch[1]], input_lang)


        print(sent)
        exit()