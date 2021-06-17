from data.data_util import extract, transfer_num_n_equation, prepare_infer_data
from src.train_and_evaluate import evaluate_tree
from src.contextual_embeddings import *
from src.models import *
from pyaichtools.pyaichtools import Reducer
from pyaichtools.pyaichtools import Converter
from pyaichtools.pyaichtools import DefaultCfg
import libcst as cst
import os
import json
import subprocess


DEBUG = False
GENERATE_DUMMY_WEIGHTS = False

weight_path = "weights/"
problem_file = "/home/agc2021/problemsheet.json"
# problem_file = "../problemsheet.json"
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
    # "pretrained_bert_path": None
    "pretrained_bert_path": './weights/embedding-80'
}

if __name__ == "__main__":

    USE_CUDA = True

    data = extract(problem_file)
    # data = extract("dummy.json")
    pairs, copy_nums = transfer_num_n_equation(data)
    input_lang, output_lang, test_pairs = prepare_infer_data(pairs, 1)


    if DEBUG:
        print("testing sample {} has been loaded".format(len(test_pairs)))

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

    reducer = Reducer(label_root_path=os.path.join(os.getcwd(), "pyaichtools", "label"))
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

    state_dicts = {
        'encoder': torch.load(os.path.join(os.getcwd(), "weights/encoder-80.pth")),
        'decoder': torch.load(os.path.join(os.getcwd(), "weights/decoder-80.pth")),
        'attention_decoder': torch.load(os.path.join(os.getcwd(), "weights/attention_decoder-80.pth")),
    }

    if DEBUG:
        print("state_dicts are successfully loaded")

    encoder.load_state_dict(state_dicts['encoder'])
    decoder.load_state_dict(state_dicts['decoder'])
    attention_decoder.load_state_dict(state_dicts['attention_decoder'])

    if DEBUG:
        print("loading state_dict to model success")

    answers = {}




    total = len(test_pairs)
    wrong_count = 0
    correct = 0

    inferrence_error = []
    wrong_answer = []
    for test_batch in test_pairs:
        idx = test_batch[-1]
        one_answer = {}
        # input_batch, input_length, operate_nums(n), embedding, encoder, decoder, attention_decoder, reducer,
        # input_lang, output_lang, num_value, num_pos(n), batch_graph(n), beam_size(n), max_length=MAX_OUTPUT_LENGTH
        try:
            test_res = evaluate_tree(test_batch[0], test_batch[1], embedding, encoder, decoder, attention_decoder, reducer,
                                     input_lang, output_lang, test_batch[2], beam_size=beam_size, num_pos=test_batch[3])
        except:
            inferrence_error.append(idx)

        QL = test_batch[2]
        NL = test_batch[4]

        QL = [eval(s) for s in QL]

        converter = Converter(DefaultCfg, debug=True)
        str_quality_list = 'QL=' + str(QL) +"\nNL=" + str(NL)
        converter.quality_list = cst.parse_module(str_quality_list)

        # if test_res == data[idx]['answer']:
        #     correct += 1
        #     # print(correct)
        # else:
        #     wrong_answer.append(idx)

        try:
            dec_seq = converter.decode(test_res)
            with open("dummy.py", "w", encoding="utf-8") as f:
                f.write(dec_seq)

            # ========================== important"
            # change python3 to python
            proc = subprocess.Popen('python dummy.py', stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, shell=True)
            result, err = proc.communicate()
            answer = result.decode('utf-8')
            # print(answer)
            if len(answer) == 0 or "Error" in answer:
                answer = "0"

            one_answer = {
                "answer": answer,
                "equation": dec_seq
            }
            # print(idx)
            # print("result=" , answer)
            os.system("rm -rf dummy.py")
        except:
            one_answer = {
                "answer": "0",
                "equation": "WRONG",
            }
            if DEBUG:
                wrong_count += 1
                # print("wrong input {}/{}".format(idx, wrong_count))
        answers[idx] = one_answer

    if DEBUG:
        with open("results/result.txt", "w", encoding="utf-8") as f:
            f.write(str(inferrence_error))
            f.write(str(wrong_answer))

    with open("answersheet.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(answers, indent=4))

    # with open()