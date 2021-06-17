from data.data_util import extract, transfer_num_n_equation, prepare_infer_data
from src.train_and_evaluate import evaluate_tree, evaluate_tree_ensemble_beam_search
from src.contextual_embeddings import *
from src.models import *
from pyaichtools.pyaichtools import Reducer
from pyaichtools.pyaichtools import Converter
from pyaichtools.pyaichtools import DefaultCfg
import libcst as cst
import os
import json
import sys
from io import StringIO
from contextlib import redirect_stdout
import pickle as pkl

weight_path = "weights/"
problem_file = "/home/agc2021/dataset/problemsheet.json"
# problem_file = "dataset/problemsheet.json"
answer_file = "answersheet.json"

MAX_OUTPUT_LENGTH = 500

batch_size = 64
embedding_size = 768
hidden_size = 512
n_epochs = 80
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2
num_workers = 20

target_epoch = 80

pretrained_model_paths = {
    "embeddings": ['./weights/embedding-{}'.format(target_epoch)],
    "encoders": ['./weights/encoder-{}.pth'.format(target_epoch)],
    "decoders": ['./weights/decoder-{}.pth'.format(target_epoch)],
    "attention_decoders": ['./weights/attention_decoder-{}.pth'.format(target_epoch)]
}

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
    "dropout_input": 0.5
}

if __name__ == "__main__":
    USE_CUDA = True
    MAX_PROBLEM_LENGTH = 1000
    MAX_NUM_MODEL = 10
    MAX_BEAM_WIDTH = 5

    data = extract(problem_file)
    pairs, copy_nums = transfer_num_n_equation(data)
    input_lang, output_lang, test_pairs = prepare_infer_data(pairs, 1)

    problem_length = len(pairs)
    num_models = int(round(MAX_NUM_MODEL * MAX_PROBLEM_LENGTH / problem_length))
    beam_width = int(round(MAX_BEAM_WIDTH * MAX_PROBLEM_LENGTH / problem_length))

    embeddings = list()
    encoders = list()
    decoders = list()
    attention_decoders = list()
    for model_i in range(len(pretrained_model_paths["embeddings"])):
        embedding = BertEncoder(pretrained_model_paths["embeddings"][model_i],
                                "cuda" if USE_CUDA else "cpu", False)
        encoder = EncoderSeq('gru', embedding_size=opt['embedding_size'], hidden_size=hidden_size,
                             n_layers=n_layers)
        decoder = DecoderRNN(opt, output_lang.n_words)
        attention_decoder = AttnUnit(opt, output_lang.n_words)

        if USE_CUDA:
            embedding.cuda()
            encoder.cuda()
            decoder.cuda()
            attention_decoder.cuda()

        encoder.load_state_dict(torch.load(pretrained_model_paths["encoders"][model_i]))
        decoder.load_state_dict(torch.load(pretrained_model_paths["decoders"][model_i]))
        attention_decoder.load_state_dict(torch.load(pretrained_model_paths["attention_decoders"][model_i]))

    reducer = Reducer(label_root_path=os.path.join(os.getcwd(), "pyaichtools", "label"))

    answers = {}

    total = len(test_pairs)
    wrong_count = 0
    correct = 0

    model_output = []
    for test_batch in test_pairs:
        idx = test_batch[-1]
        one_answer = {}
        try:
            test_res = evaluate_tree_ensemble_beam_search(test_batch[0], test_batch[1], [],
                                                          embeddings, encoders, decoders, attention_decoders,
                                                          input_lang, output_lang, test_batch[2], beam_size=beam_size)
        except:
            test_res = "Fail"

        QL = test_batch[2]
        NL = test_batch[4]

        QL = [eval(s) for s in QL]

        converter = Converter(DefaultCfg, debug=True)
        str_quality_list = 'QL=' + str(QL) +"\nNL=" + str(NL)
        converter.quality_list = cst.parse_module(str_quality_list)

        if not test_res == "Fail":
            try:
                # print(test_res)
                dec_seq = converter.decode(test_res)
                f = StringIO()
                with redirect_stdout(f):
                    exec(dec_seq)
                answer = f.getvalue()

                if len(answer) == 0 or "Error" in answer:
                    answer = "0"

                one_answer = {
                    "answer": answer,
                    "equation": dec_seq
                }

            except:
                print(sys.exc_info())
                one_answer = {
                    "answer": "0",
                    "equation": "WRONG",
                }

            answers[idx] = one_answer
        else:
            answers[idx] = {
                "answer": "0",
                "equation": "fail"
            }

    with open(answer_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(answers, indent=4))
