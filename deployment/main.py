import json
import torch
import os
import codecs
from data_util import read_json_mac, extract, transfer_num_n_equation, prepare_infer_data
from model_util import load_model
from src.pre_data import transfer_num
from src.helper import index_batch_to_words, sentence_from_indexes
from src.train_and_evaluate import evaluate_tree
from src.contextual_embeddings import *
from src.models import *
from pyaichtools import Reducer

DEBUG = True

weight_path = "weights/"
data_path = ""
problem_file = "problemsheet.json"
# problem_file = "/home/agc2021/dataset/problemsheet.json"
answer_file = "answersheet.json"

MAX_OUTPUT_LENGTH = 100
GENERATE_DUMMY_WEIGHTS = False

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

opt = {
    "rnn_size": hidden_size, # RNN hidden size (default 300)
    # "num_layers": 2, # RNN # of layer (default 1)
    "dropout_de_in": 0.1,
    "dropout_de_out": 0.3,
    "dropout_for_predict": 0.1,
    "dropoutagg": 0,
    "learningRate": learning_rate, # default 1.0e-3
    "init_weight": 0.08,
    "grad_clip": 5,
    "separate_attention": False,

    # for BERT
    "bert_learningRate": learning_rate * 1e-2,
    "embedding_size": 768,
    "dropout_input": 0.5,
    "pretrained_bert_path": None
    # "pretrained_bert_path": './electra_model'
}

if __name__ == "__main__":

    USE_CUDA = True

    data = extract(problem_file)
    pairs, copy_nums = transfer_num_n_equation(data)
    input_lang, output_lang, test_pairs = prepare_infer_data(pairs, 1)

    if DEBUG:
        print("testing sample {} has been loaded".format(len(test_pairs)))

    # for d in data:
    #     print(" ".join(data[d]['question']))
    #     input()
    # exit()

    embedding = BertEncoder(opt["pretrained_bert_path"], "cuda" if USE_CUDA else "cpu", False)
    encoder = EncoderSeq('gru', embedding_size=opt['embedding_size'], hidden_size=hidden_size,
                         n_layers=n_layers)
    decoder = DecoderRNN(opt, output_lang.n_words)
    attention_decoder = AttnUnit(opt, output_lang.n_words)

    if USE_CUDA:
        embedding.cuda()
        encoder.cuda()
        decoder.cuda()
        attention_decoder.cuda()

    reducer = Reducer(label_root_path="data")
    if DEBUG:
        print("reducer loaded")

    if GENERATE_DUMMY_WEIGHTS:
        # -------- test code ---------- #
        if DEBUG:
            print("generate dummy weights")
        state_dict = {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "attention_decoder": attention_decoder.state_dict()
        }
        torch.save(state_dict, "weights/state_dicts.pth")

    state_dicts = torch.load("weights/state_dicts.pth")
    if DEBUG:
        print("state_dicts are successfully loaded")

    for test_batch in test_pairs:
        # sent = index_batch_to_words([test_batch[0]], [test_batch[1]], input_lang)
        # test_res = evaluate_tree(test_batch[0], test_batch[1], embedding, encoder, decoder,
        #                          attention_decoder, input_lang, test_batch[2])

        # input_batch, input_length, operate_nums(n), embedding, encoder, decoder, attention_decoder, reducer,
        # input_lang, output_lang, num_value, num_pos(n), batch_graph(n), beam_size(n), max_length=MAX_OUTPUT_LENGTH
        test_res = evaluate_tree(test_batch[0], test_batch[1], embedding, encoder, decoder, attention_decoder, reducer,
                                 input_lang, output_lang, test_batch[2], beam_size=beam_size)

        print(test_res)
        exit()