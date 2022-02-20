# coding: utf-8
from numpy.lib.function_base import copy
from src.train_and_evaluate import *
from src.models import *
from src.contextual_embeddings import *
import time
import torch.optim
from src.expressions_transfer import *
import json
import sympy
import os
import numpy as np
from sympy.parsing.sympy_parser import parse_expr
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold
from nltk.translate.bleu_score import sentence_bleu
from torch.utils.data import DataLoader
from src.pre_data import TrainDataset, my_collate
from pyaichtools import Reducer

def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file


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

log_path = "logs/{}".format("NoSepAtt_Max")
num_folds = 12
target_epoch = 120
target_folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
num_workers = 20
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
optimizer_patience = 10
random_seed = 777

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

if not os.path.exists("logs"):
    try:
        os.mkdir("logs")
    except OSError:
        pass

def get_new_fold(data,pairs,group,mask):
    new_fold = []
    for item,pair,g,mask_ele in zip(data, pairs, group, mask):
        pair = list(pair)
        pair.append(g['group_num'])
        pair.append(mask_ele)
        pair = tuple(pair)
        new_fold.append(pair)
    return new_fold

def change_num(num):
    new_num = []
    for item in num:
        if '/' in item:
            new_str = item.split(')')[0]
            new_str = new_str.split('(')[1]
            a = float(new_str.split('/')[0])
            b = float(new_str.split('/')[1])
            value = a/b
            new_num.append(value)
        elif '%' in item:
            value = float(item[0:-1])/100
            new_num.append(value)
        else:
            new_num.append(float(item))
    return new_num

def convert_to_string(idx_list, output_lang):
    w_list = []
    for i in range(len(idx_list)):
        w_list.append(output_lang.index2word[int(idx_list[i])])
    return " ".join(w_list)


def is_all_same(c1, c2, output_lang):
    all_same = False
    if len(c1) == len(c2):
        all_same = True
        for j in range(len(c1)):
            if c1[j] != c2[j]:
                all_same = False
                break
    if all_same == False:
        if is_solution_same(c1, c2, output_lang):
            return True
        return False
    else:
        return True


def is_solution_same(i1, i2, output_lang):
    c1 = " ".join([output_lang.index2word[x] for x in i1])
    c2 = " ".join([output_lang.index2word[x] for x in i2])
    if ('=' not in c1) or ('=' not in c2):
        return False
    elif ('<U>' in c1) or ('<U>' in c2):
        return False
    else:
        try:
            s1 = c1.split('=')
            s2 = c2.split('=')
            eq1 = []
            eq2 = []
            x = sympy.Symbol('x')
            eq1.append(parse_expr(s1[0]))
            eq1.append(parse_expr(s1[1]))
            eq2.append(parse_expr(s2[0]))
            eq2.append(parse_expr(s2[1]))
            res1 = sympy.solve(sympy.Eq(eq1[0], eq1[1]), x)
            res2 = sympy.solve(sympy.Eq(eq2[0], eq2[1]), x)

            if not res1 or not res2:
                return False

            return res1[0] == res2[0]

        except BaseException:
            print(c1)
            print(c2)
            return False

def compute_accuracy(candidate_list, reference_list, output_lang):
    if len(candidate_list) != len(reference_list):
        print("candidate list has length {}, reference list has length {}\n".format(len(candidate_list),
                                                                                    len(reference_list)))

    len_min = min(len(candidate_list), len(reference_list))
    c = 0
    for i in range(len_min):
        # print "length:", len_min

        if is_all_same(candidate_list[i], reference_list[i], output_lang):
            # print "{}->True".format(i)
            c = c + 1
        else:
            # print "{}->False".format(i)
            pass
    return c / float(len_min)

def compute_tree_accuracy(candidate_list_, reference_list_, output_lang):
    candidate_list = []

    for i in range(len(candidate_list_)):
        candidate_list.append(candidate_list_[i])
    reference_list = []
    for i in range(len(reference_list_)):
        reference_list.append(reference_list_[i])
    return compute_accuracy(candidate_list, reference_list, output_lang)

def ref_flatten(ref, output_lang):
    flattened_ref = list()
    for x in ref:
        if type(x) == type(list()):
            flattened_ref.append(output_lang.word2index["<IS>"])
            flattened_ref += ref_flatten(x, output_lang)
            flattened_ref.append(output_lang.word2index["<IE>"])
        else:
            flattened_ref.append(x)

    return flattened_ref

data = load_mawps_data("data/custom_dummy.json")
group_data = read_json("data/new_MAWPS_processed.json")
mask_data_ele = read_json("data/mask_processed.json")
# dummy data has same Ground truth code, so repeat mask_data
mask_data = []
for i in range(len(data)):
    mask_data.append(copy.deepcopy(mask_data_ele))

reducer = Reducer(label_root_path="data")

pairs, generate_nums, copy_nums = transfer_english_num(data)

# temp_pairs = []
# for p in pairs:
#     temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
# pairs = temp_pairs

#train_fold, test_fold, valid_fold = get_train_test_fold(ori_path,prefix,data,pairs,group_data)
pairs = get_new_fold(data, pairs, group_data, mask_data)

fold_size = int(len(pairs) * (1.0 / num_folds))
fold_pairs = []
for split_fold in range(4):
    fold_start = fold_size * split_fold
    fold_end = fold_size * (split_fold + 1)
    fold_pairs.append(pairs[fold_start:fold_end])
fold_pairs.append(pairs[(fold_size * 4):])
# whole_fold = fold_pairs
# random.shuffle(whole_fold)

pretrained_model_paths = {
    "embeddings": ['../weights/FINAL/Fold_{:02d}/embedding-{}'.format(f_i, target_epoch) for f_i in range(num_folds)],
    "encoders": ['../weights/FINAL/Fold_{:02d}/encoder-{}.pth'.format(f_i, target_epoch) for f_i in range(num_folds)],
    "decoders": ['../weights/FINAL/Fold_{:02d}/decoder-{}.pth'.format(f_i, target_epoch) for f_i in range(num_folds)],
    "attention_decoders": ['../weights/FINAL/Fold_{:02d}/attention_decoder-{}.pth'.format(f_i, target_epoch)
                           for f_i in range(num_folds)]}

best_accuracies = list()
best_bleu_scores = list()
for model_i, fold in enumerate(target_folds):

    pairs_tested = []
    pairs_trained = []
    for fold_t in range(num_folds):
        if fold_t == fold:
            pairs_tested += fold_pairs[fold_t]
        else:
            pairs_trained += fold_pairs[fold_t]

    input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 1, generate_nums,
                                                                    copy_nums, tree=False)

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

    embedding_optimizer = torch.optim.Adam(embedding.parameters(), lr=opt['bert_learningRate'], weight_decay=weight_decay)
    encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=opt["learningRate"], weight_decay=weight_decay)
    attention_decoder_optimizer = torch.optim.AdamW(attention_decoder.parameters(), lr=opt["learningRate"],
                                                    weight_decay=weight_decay)

    embedding_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(embedding_optimizer, 'min', patience=optimizer_patience)

    encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer,
                                                                   'min', patience=optimizer_patience)
    decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer,
                                                                   'min', patience=optimizer_patience)
    attention_decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(attention_decoder_optimizer,
                                                                             'min', patience=optimizer_patience)

    generate_num_ids = []

    print("Fold:", fold + 1)
    dataset = TrainDataset(test_pairs, input_lang, output_lang, USE_CUDA)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=my_collate, pin_memory=True, num_workers=num_workers)

    reference_list = list()
    candidate_list = list()
    bleu_scores = list()
    for test_batch in test_pairs:
        # batch_graph = get_single_example_graph(test_batch[0], test_batch[1], test_batch[7], test_batch[4], test_batch[5])
        # test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, embedding, encoder, decoder, attention_decoder, reducer,
        #                          input_lang, output_lang, test_batch[4], test_batch[5], batch_graph, beam_size=beam_size)
        test_res = evaluate_tree_ensemble_beam_search(test_batch[0], test_batch[1], generate_num_ids,
                                                      [embedding], [encoder], [decoder], [attention_decoder],
                                                      input_lang, output_lang, test_batch[4], beam_size=beam_size)
        reference = test_batch[2]
        candidate = [int(c) for c in test_res]
        reference = ref_flatten(reference, output_lang)

        ref_str = convert_to_string(reference, output_lang)
        cand_str = convert_to_string(candidate, output_lang)

        reference_list.append(reference)
        candidate_list.append(candidate)

        bleu_score = sentence_bleu([reference], candidate, weights=(0.5, 0.5))
        bleu_scores.append(bleu_score)

    accuracy = compute_tree_accuracy(candidate_list, reference_list, output_lang)
    bleu_scores = np.mean(bleu_scores)

    print("validation_accuracy:", accuracy)
    print("validation_bleu_score:", bleu_scores)
    print("--------------------------------")

    best_accuracies.append(accuracy)
    best_bleu_scores.append(bleu_scores)

for fold_i in range(num_folds):
    print("-" * 50)
    print("Fold_{:01d} Best Accuracy: {:.5f}".format(fold_i + 1, best_accuracies[fold_i]))
    print("Fold_{:01d} Best BLEU Score: {:.5f}".format(fold_i + 1, best_bleu_scores[fold_i]))
print("-" * 50)
print("Average Best Accuracy: {:.5f}".format(np.mean(best_accuracies)))
print("Average Best BLEU Score: {:.5f}".format(np.mean(best_bleu_scores)))
print("-" * 50)
