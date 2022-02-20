# coding: utf-8
import random

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
n_epochs = 120
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2
ori_path = './data/'
prefix = '23k_processed.json'

opt = {
    "rnn_size": hidden_size, # RNN hidden size (default 300)
    "dropout_de_in": 0.1,
    "dropout_de_out": 0.3, # default 0.3
    "dropout_for_predict": 0.1,
    "dropoutagg": 0,
    "learningRate": learning_rate, # default 1.0e-3
    "init_weight": 0.08,
    "grad_clip": 5,
    "separate_attention": False,

    # for BERT
    "bert_learningRate": learning_rate * 1e-2,
    # "bert_learningRate": learning_rate * 1.0e-1,
    "embedding_size": 768,
    "dropout_input": 0.5,
    "pretrained_bert_path": None
    # "pretrained_bert_path": './electra_model'
}

num_folds = 12
target_folds = list(range(num_folds))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
optimizer_patience = 20
num_workers = 20
random_seed = 777

target_epoch = 120

criterion = torch.nn.NLLLoss(size_average=False, ignore_index=0)

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

def get_new_fold(data,pairs):
    new_fold = []
    for item,pair in zip(data, pairs):
        pair = list(pair)
        pair.append([1, 2])
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
            flattened_ref.append(output_lang.word2index["<E>"])
        else:
            flattened_ref.append(x)

    return flattened_ref

data = load_mawps_data("data/koco_04_new.json")
group_data = read_json("data/new_MAWPS_processed.json")

pairs, generate_nums, copy_nums = transfer_english_num(data)

pairs = get_new_fold(data, pairs)
random.shuffle(pairs)

fold_size = int(len(pairs) * (1.0 / num_folds))
fold_pairs = []
for split_fold in range(num_folds):
    fold_start = fold_size * split_fold
    fold_end = fold_size * (split_fold + 1)
    fold_pairs.append(pairs[fold_start:fold_end])
fold_pairs.append(pairs[(fold_size * (num_folds)):])

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

    # Initialize models
    embedding = BertEncoder(opt["pretrained_bert_path"], "cuda" if USE_CUDA else "cpu", False)
    embedding.to("cpu")
    encoder = EncoderSeq('gru', embedding_size=opt['embedding_size'], hidden_size=hidden_size,
                         n_layers=n_layers)

    decoder = DecoderRNN(opt, output_lang.n_words)
    attention_decoder = AttnUnit(opt, output_lang.n_words)

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

    if USE_CUDA:
        embedding.cuda()
        encoder.cuda()
        decoder.cuda()
        attention_decoder.cuda()

    generate_num_ids = []

    fold_best_accuracy = -1
    fold_best_bleu = -1

    print("fold:", fold + 1)

    reference_list = list()
    candidate_list = list()
    bleu_scores = list()
    for test_batch in test_pairs:
        batch_graph = get_single_example_graph(test_batch[0], test_batch[1],
                                               test_batch[7], test_batch[4], test_batch[5])
        test_res = evaluate_tree_ensemble_beam_search(
            test_batch[0], test_batch[1], generate_num_ids,
            [embedding], [encoder], [decoder], [attention_decoder],
            input_lang, output_lang, test_batch[4], test_batch[5], batch_graph,
            beam_size=beam_size)

        reference = test_batch[2]
        candidate = [int(c) for c in test_res]

        reference = ref_flatten(reference, output_lang)

        ref_str = convert_to_string(reference, output_lang)
        cand_str = convert_to_string(candidate, output_lang)

        reference_list.append(reference)
        candidate_list.append(candidate)

        bleu_score = sentence_bleu([reference], candidate, weights=(0.5, 0.5))
        bleu_scores.append(bleu_score)
    reference = [output_lang.index2word[x] for x in reference]
    candidate = [output_lang.index2word[x] for x in candidate]
    accuracy = compute_tree_accuracy(candidate_list, reference_list, output_lang)
    bleu_scores = np.mean(bleu_scores)
    fold_best_accuracy = accuracy
    fold_best_bleu = bleu_scores

    print("=" * 90)
    print("Fold_{:01d} Accuracy: {:.5f}".format(fold + 1, accuracy))
    print("Fold_{:01d} BLEU Score: {:.5f}".format(fold + 1, bleu_scores))

    best_accuracies.append(fold_best_accuracy)
    best_bleu_scores.append(fold_best_bleu)

for fold_i in range(num_folds):
    print("-" * 90)
    print("Fold_{:01d} Best Accuracy: {:.5f}".format(fold_i + 1, best_accuracies[fold_i]))
    print("Fold_{:01d} Best BLEU Score: {:.5f}".format(fold_i + 1, best_bleu_scores[fold_i]))
print("-" * 90)
print("Average Best Accuracy: {:.5f}".format(np.mean(best_accuracies)))
print("Average Best BLEU Score: {:.5f}".format(np.mean(best_bleu_scores)))
print("-" * 90)
