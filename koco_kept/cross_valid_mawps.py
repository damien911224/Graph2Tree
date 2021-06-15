# coding: utf-8
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
    "separate_attention": True,

    # for BERT
    "bert_learningRate": learning_rate * 1e-2,
    "embedding_size": 768,
    "dropout_input": 0.5,
    "pretrained_bert_path": None
}

log_path = "logs/{}".format("SepAtt_EncoderSplit")
num_folds = 5
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

def get_new_fold(data,pairs,group):
    new_fold = []
    for item,pair,g in zip(data, pairs, group):
        pair = list(pair)
        pair.append(g['group_num'])
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

data = load_mawps_data("data/mawps_combine.json")
group_data = read_json("data/new_MAWPS_processed.json")

pairs, generate_nums, copy_nums = transfer_english_num(data)

# temp_pairs = []
# for p in pairs:
#     temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
# pairs = temp_pairs

#train_fold, test_fold, valid_fold = get_train_test_fold(ori_path,prefix,data,pairs,group_data)
new_fold = get_new_fold(data, pairs, group_data)
pairs = new_fold

fold_size = int(len(pairs) * (1.0 / num_folds))
fold_pairs = []
for split_fold in range(4):
    fold_start = fold_size * split_fold
    fold_end = fold_size * (split_fold + 1)
    fold_pairs.append(pairs[fold_start:fold_end])
fold_pairs.append(pairs[(fold_size * 4):])
whole_fold = fold_pairs
# random.shuffle(whole_fold)

best_accuracies = list()
best_bleu_scores = list()
for fold in range(num_folds):
    fold_log_folder = os.path.join(log_path, "Fold_{:02d}".format(fold + 1))
    fold_weight_folder = os.path.join(fold_log_folder, "weights")
    try:
        os.makedirs(fold_weight_folder)
    except OSError:
        pass
    writer = SummaryWriter(fold_log_folder)

    pairs_tested = []
    pairs_trained = []
    for fold_t in range(num_folds):
        if fold_t == fold:
            pairs_tested += fold_pairs[fold_t]
        else:
            pairs_trained += fold_pairs[fold_t]

    # ===============changed=================
    # input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,
    #                                                                     copy_nums, tree=False)
    input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 1, generate_nums,
                                                                    copy_nums, tree=False)

    #print('train_pairs[0]')
    #print(train_pairs[0])
    #exit()
    # ===============changed=================
    if True:
        # Initialize models
        embedding = BertEncoder(opt["pretrained_bert_path"], "cuda:0", False)
        # embedding = BertEncoder("bert-base-uncased", "cuda:0", False)
    else:
        embedding = Embedding(None, input_lang, input_size=input_lang.n_words, embedding_size=opt['embedding_size'], dropout=opt['dropout_input'])
    encoder = EncoderSeq('gru', embedding_size=opt['embedding_size'], hidden_size=hidden_size,
                         n_layers=n_layers)
    ##################################################################################################
    # Initialize models
    # encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
    #                      n_layers=n_layers)
    # predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
    #                      input_size=len(generate_nums))
    # generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
    #                         embedding_size=embedding_size)
    # merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)

    decoder = DecoderRNN(opt, output_lang.n_words)
    attention_decoder = AttnUnit(opt, output_lang.n_words)
    # the embedding layer is  only for generated number embeddings, operators, and paddings

    embedding_optimizer = torch.optim.Adam(embedding.parameters(), lr=opt['bert_learningRate'], weight_decay=weight_decay)
    encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=opt["learningRate"], weight_decay=weight_decay)
    attention_decoder_optimizer = torch.optim.AdamW(attention_decoder.parameters(), lr=opt["learningRate"],
                                                    weight_decay=weight_decay)

    # encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=20, gamma=0.5)
    # decoder_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=20, gamma=0.5)
    # attention_decoder_scheduler = torch.optim.lr_scheduler.StepLR(attention_decoder_optimizer, step_size=20, gamma=0.5)

    # ===============changed=================
    # embedding_scheduler = torch.optim.lr_scheduler.StepLR(embedding_optimizer, step_size=20, gamma=0.5)
    embedding_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(embedding_optimizer, 'min', patience=optimizer_patience)

    encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer,
                                                                   'min', patience=optimizer_patience)
    decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer,
                                                                   'min', patience=optimizer_patience)
    attention_decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(attention_decoder_optimizer,
                                                                             'min', patience=optimizer_patience)

    # Move models to GPU
    if USE_CUDA:
        embedding.cuda()
        encoder.cuda()
        decoder.cuda()
        attention_decoder.cuda()

    generate_num_ids = []
    # for num in generate_nums:
    #     generate_num_ids.append(output_lang.word2index[num])

    fold_best_accuracy = -1
    fold_best_bleu = -1
    for epoch in range(n_epochs):
        print("fold:", fold + 1)
        print("epoch:", epoch + 1)

        start = time.time()

        train_loss_total = 0
        input_batches, input_lengths, output_batches, output_lengths, nums_batches, \
        num_stack_batches, num_pos_batches, num_size_batches, \
        num_value_batches, graph_batches = prepare_train_batch(train_pairs, batch_size)
        for idx in range(len(input_lengths)):
            train_loss = train_tree(
                input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                num_stack_batches[idx], num_size_batches[idx], num_value_batches[idx], generate_num_ids, embedding, encoder, decoder, attention_decoder,
                embedding_optimizer, encoder_optimizer, decoder_optimizer, attention_decoder_optimizer,
                input_lang, output_lang, num_pos_batches[idx], graph_batches[idx])
            train_loss_total += train_loss.detach().cpu().numpy()
        train_loss_total = train_loss_total / len(input_lengths)

        val_loss_total = 0
        input_batches, input_lengths, output_batches, output_lengths, nums_batches, \
        num_stack_batches, num_pos_batches, num_size_batches, \
        num_value_batches, graph_batches = prepare_train_batch(test_pairs, batch_size)
        for idx in range(len(input_lengths)):
            val_loss = val_tree(
                input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                num_stack_batches[idx], num_size_batches[idx], num_value_batches[idx], generate_num_ids, embedding,
                encoder, decoder, attention_decoder,
                embedding_optimizer, encoder_optimizer, decoder_optimizer, attention_decoder_optimizer,
                input_lang, output_lang, num_pos_batches[idx], graph_batches[idx])
            val_loss_total += val_loss.detach().cpu().numpy()
        val_loss_total = val_loss_total / len(input_lengths)

        # if epoch % 2 == 0 or epoch > n_epochs - 5:
        # value_ac = 0
        # equation_ac = 0
        # eval_total = 0
        # start = time.time()
        reference_list = list()
        candidate_list = list()
        bleu_scores = list()
        for test_batch in test_pairs:
            #print(test_batch)
            batch_graph = get_single_example_graph(test_batch[0], test_batch[1], test_batch[7], test_batch[4], test_batch[5])
            test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, embedding, encoder, decoder, attention_decoder,
                                     input_lang, output_lang, test_batch[4], test_batch[5], batch_graph, beam_size=beam_size)
            reference = test_batch[2]
            # val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
            # if val_ac:
            #     value_ac += 1
            # if equ_ac:
            #     equation_ac += 1
            # eval_total += 1
            candidate = [int(c) for c in test_res]

            # num_left_paren = sum(1 for c in candidate if output_lang.index2word[int(c)] == "(")
            # num_right_paren = sum(1 for c in candidate if output_lang.index2word[int(c)] == ")")
            # diff = num_left_paren - num_right_paren
            #
            # if diff > 0:
            #     for i in range(diff):
            #         candidate.append(output_lang.index2word[")"])
            # elif diff < 0:
            #     candidate = candidate[:diff]

            reference = ref_flatten(reference, output_lang)

            ref_str = convert_to_string(reference, output_lang)
            cand_str = convert_to_string(candidate, output_lang)

            reference_list.append(reference)
            candidate_list.append(candidate)

            bleu_score = sentence_bleu([reference], candidate, weights=(0.5, 0.5))
            bleu_scores.append(bleu_score)
        # print(equation_ac, value_ac, eval_total)
        # print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
        # print("testing time", time_since(time.time() - start))
        # print("------------------------------------------------------")
        accuracy = compute_tree_accuracy(candidate_list, reference_list, output_lang)
        bleu_scores = np.mean(bleu_scores)

        encoder_scheduler.step(val_loss_total)
        decoder_scheduler.step(val_loss_total)
        attention_decoder_scheduler.step(val_loss_total)

        torch.save(embedding.state_dict(), os.path.join(fold_weight_folder, "embedding-{}.pth".format(epoch + 1)))
        torch.save(encoder.state_dict(), os.path.join(fold_weight_folder, "encoder-{}.pth".format(epoch + 1)))
        torch.save(decoder.state_dict(), os.path.join(fold_weight_folder, "decoder-{}.pth".format(epoch + 1)))
        torch.save(attention_decoder.state_dict(),
                   os.path.join(fold_weight_folder, "attention_decoder-{}.pth".format(epoch + 1)))
        # if epoch == n_epochs - 1:
        #     best_acc_fold.append(accuracy)

        if accuracy >= fold_best_accuracy:
            fold_best_accuracy = accuracy

        if bleu_scores >= fold_best_bleu:
            fold_best_bleu = bleu_scores

        current_lr = encoder_optimizer.param_groups[0]['lr']
        writer.add_scalars("Loss", {"train": train_loss_total}, epoch + 1)
        writer.add_scalars("Loss", {"val": val_loss_total}, epoch + 1)
        writer.add_scalars("Accuracy", {"val": accuracy}, epoch + 1)
        writer.add_scalars("BLEU Score", {"val": bleu_scores}, epoch + 1)
        writer.add_scalar("Learning Rate", current_lr, epoch + 1)

        print("train_loss:", train_loss_total)
        print("validation_loss:", val_loss_total)
        print("validation_accuracy:", accuracy)
        print("validation_bleu_score:", bleu_scores)
        print("current_learning_rate:", current_lr)
        print("training time:", time_since(time.time() - start))
        print("--------------------------------")
    best_accuracies.append(fold_best_accuracy)
    best_bleu_scores.append(fold_best_bleu)

for fold_i in range(num_folds):
    print("-" * 50)
    print("Fold_{:01d} Best Accuracy: {:.5f}".format(fold_i + 1, best_accuracies[fold_i]))
    print("Fold_{:01d} Best BLEU Score: {:.5f}".format(fold_i + 1, best_bleu_scores[fold_i]))
print("-" * 50)
print("Average Best Accuracy: {:.5f}".format(np.mean(best_accuracies)))
print("Average Best BLEU Score: {:.5f}".format(np.mean(best_bleu_scores)))
print("-" * 50)
