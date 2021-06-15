import json
import torch
import os
import codecs
from data_util import read_json_mac, extract, transfer_num_n_equation, prepare_infer_data, CustomDataSet
from model_util import load_model
from src.pre_data import transfer_num
from src.helper import index_batch_to_words, sentence_from_indexes
from src.train_and_evaluate import evaluate_tree
from cross_valid_mawps import opt
from src.contextual_embeddings import *
from src.models import *

weight_path = ""
data_path = ""
problem_file = "problemsheet.json"
answer_file = "answersheet.json"

MAX_OUTPUT_LENGTH = 100

batch_size = 64
# embedding_size = 128
# ===============changed=================
embedding_size = 768
# =======================================
hidden_size = 512
n_epochs = 80
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2
ori_path = './data/'
prefix = '23k_processed.json'
num_workers = 20

if __name__ == "__main__":

    USE_CUDA = True

    data = extract(problem_file)
    pairs, copy_nums = transfer_num_n_equation(data)
    input_lang, output_lang, test_pairs = prepare_infer_data(pairs, 5)

    embedding = BertEncoder(opt["pretrained_bert_path"], "cuda" if USE_CUDA else "cpu", False)
    encoder = EncoderSeq('gru', embedding_size=opt['embedding_size'], hidden_size=hidden_size,
                         n_layers=n_layers)
    decoder = DecoderRNN(opt, output_lang.n_words)
    attention_decoder = AttnUnit(opt, output_lang.n_words)

    # -------- test code ---------- #
    state_dict = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "attention_decoder": attention_decoder.state_dict()
    }

    for test_batch in test_pairs:
        sent = index_batch_to_words([test_batch[0]], [test_batch[1]], input_lang)
        test_res = evaluate_tree(test_batch[0], test_batch[1], embedding, encoder, decoder,
                                 attention_decoder, input_lang, test_batch[2])

        print(test_batch)

        print(sent)
        exit()

