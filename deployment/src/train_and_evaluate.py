from src.masked_cross_entropy import *
from src.pre_data import *
from src.expressions_transfer import *
from src.models import *
import math
import torch
import torch.optim
import torch.nn.functional as f
import time
import random
from torch import nn

from src.helper import index_batch_to_words, sort_by_len

MAX_OUTPUT_LENGTH = 200
MAX_INPUT_LENGTH = 500
USE_CUDA = torch.cuda.is_available()


class Beam:  # the class save the beam node
    def __init__(self, score, input_var, hidden, all_output):
        self.score = score
        self.input_var = input_var
        self.hidden = hidden
        self.all_output = all_output


def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)


def generate_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums, generate_nums,
                       english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + \
                      [word2index["("]] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] in generate_nums:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] == word2index["("]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["("]] + generate_nums
            elif decoder_input[i] == word2index[")"]:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + [word2index["("]] + generate_nums
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + \
                      [word2index["["], word2index["("]] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [word2index["]"], word2index[")"], word2index["+"],
                        word2index["-"], word2index["/"], word2index["^"],
                        word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] == word2index["["] or decoder_input[i] == word2index["("]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["("]] + generate_nums
            elif decoder_input[i] == word2index[")"]:
                res += [word2index["]"], word2index[")"], word2index["+"],
                        word2index["-"], word2index["/"], word2index["^"],
                        word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["]"]:
                res += [word2index["+"], word2index["*"], word2index["-"], word2index["/"], word2index["EOS"]]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"],
                                      word2index["*"], word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["["], word2index["("]] + generate_nums
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_pre_tree_seq_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums,
                                    generate_nums, english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                      [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                      [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"],
                        word2index["^"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
                                      word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_post_tree_seq_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums,
                                     generate_nums, english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums +\
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
                                      word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"],
                        word2index["EOS"]
                        ]
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_tree_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    target_input = copy.deepcopy(target)
    for i in range(len(target)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
        if target_input[i] >= num_start:
            target_input[i] = 0
    return torch.LongTensor(target), torch.LongTensor(target_input)


def generate_decoder_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    if USE_CUDA:
        decoder_output = decoder_output.cpu()
    for i in range(target.size(0)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
    return target


def mask_num(encoder_outputs, decoder_input, embedding_size, nums_start, copy_nums, num_pos):
    # mask the decoder input number and return the mask tensor and the encoder position Hidden vector
    up_num_start = decoder_input >= nums_start
    down_num_end = decoder_input < (nums_start + copy_nums)
    num_mask = up_num_start == down_num_end
    num_mask_encoder = num_mask < 1
    num_mask_encoder = num_mask_encoder.unsqueeze(1)  # ByteTensor size: B x 1
    repeat_dims = [1] * num_mask_encoder.dim()
    repeat_dims[1] = embedding_size
    num_mask_encoder = num_mask_encoder.repeat(*repeat_dims)  # B x 1 -> B x Decoder_embedding_size

    all_embedding = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_embedding.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    indices = decoder_input - nums_start
    indices = indices * num_mask.long()  # 0 or the num pos in sentence
    indices = indices.tolist()
    for k in range(len(indices)):
        indices[k] = num_pos[k][indices[k]]
    indices = torch.LongTensor(indices)
    if USE_CUDA:
        indices = indices.cuda()
    batch_size = decoder_input.size(0)
    sen_len = encoder_outputs.size(0)
    batch_num = torch.LongTensor(range(batch_size))
    batch_num = batch_num * sen_len
    if USE_CUDA:
        batch_num = batch_num.cuda()
    indices = batch_num + indices
    num_encoder = all_embedding.index_select(0, indices)
    return num_mask, num_encoder, num_mask_encoder


def out_equation(test, output_lang, num_list, num_stack=None):
    test = test[:-1]
    max_index = len(output_lang.index2word) - 1
    test_str = ""
    for i in test:
        if i < max_index:
            c = output_lang.index2word[i]
            if c == "^":
                test_str += "**"
            elif c == "[":
                test_str += "("
            elif c == "]":
                test_str += ")"
            elif c[0] == "N":
                if int(c[1:]) >= len(num_list):
                    return None
                x = num_list[int(c[1:])]
                if x[-1] == "%":
                    test_str += "(" + x[:-1] + "/100" + ")"
                else:
                    test_str += x
            else:
                test_str += c
        else:
            if len(num_stack) == 0:
                print(test_str, num_list)
                return ""
            n_pos = num_stack.pop()
            test_str += num_list[n_pos[0]]
    return test_str


def compute_prefix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def compute_postfix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_postfix_expression(test) - compute_postfix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def compute_result(test_res, test_tar, output_lang, num_list, num_stack):
    if len(num_stack) == 0 and test_res == test_tar:
        return True, True
    test = out_equation(test_res, output_lang, num_list)
    tar = out_equation(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    if test is None:
        return False, False
    if test == tar:
        return True, True
    try:
        if abs(eval(test) - eval(tar)) < 1e-4:
            return True, False
        else:
            return False, False
    except:
        return False, False


def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, hidden_size):
    indices = list()
    sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices)
    masked_index = torch.ByteTensor(masked_index)
    masked_index = masked_index.view(batch_size, num_size, hidden_size)
    if USE_CUDA:
        indices = indices.cuda()
        masked_index = masked_index.cuda()
    all_outputs = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    return all_num.masked_fill_(masked_index, 0.0)


def train_attn(input_batch, input_length, target_batch, target_length, num_batch, nums_stack_batch, copy_nums,
               generate_nums, encoder, decoder, encoder_optimizer, decoder_optimizer, output_lang, clip=0,
               use_teacher_forcing=1, beam_size=1, english=False):
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_start = output_lang.n_words - copy_nums - 2
    unk = output_lang.word2index["UNK"]
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    target = torch.LongTensor(target_batch).transpose(0, 1)

    batch_size = len(input_length)

    encoder.train()
    decoder.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_var, input_length, None)

    # Prepare input and output variables
    decoder_input = torch.LongTensor([output_lang.word2index["SOS"]] * batch_size)

    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    max_target_length = max(target_length)
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size)

    # Move new Variables to CUDA
    if USE_CUDA:
        all_decoder_outputs = all_decoder_outputs.cuda()

    if random.random() < use_teacher_forcing:
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            if USE_CUDA:
                decoder_input = decoder_input.cuda()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            all_decoder_outputs[t] = decoder_output
            decoder_input = generate_decoder_input(
                target[t], decoder_output, nums_stack_batch, num_start, unk)
            target[t] = decoder_input
    else:
        beam_list = list()
        score = torch.zeros(batch_size)
        if USE_CUDA:
            score = score.cuda()
        beam_list.append(Beam(score, decoder_input, decoder_hidden, all_decoder_outputs))
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            beam_len = len(beam_list)
            beam_scores = torch.zeros(batch_size, decoder.output_size * beam_len)
            all_hidden = torch.zeros(decoder_hidden.size(0), batch_size * beam_len, decoder_hidden.size(2))
            all_outputs = torch.zeros(max_target_length, batch_size * beam_len, decoder.output_size)
            if USE_CUDA:
                beam_scores = beam_scores.cuda()
                all_hidden = all_hidden.cuda()
                all_outputs = all_outputs.cuda()

            for b_idx in range(len(beam_list)):
                decoder_input = beam_list[b_idx].input_var
                decoder_hidden = beam_list[b_idx].hidden

                rule_mask = generate_rule_mask(decoder_input, num_batch, output_lang.word2index, batch_size,
                                               num_start, copy_nums, generate_nums, english)
                if USE_CUDA:
                    rule_mask = rule_mask.cuda()
                    decoder_input = decoder_input.cuda()

                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, seq_mask)

                score = f.log_softmax(decoder_output, dim=1) + rule_mask
                beam_score = beam_list[b_idx].score
                beam_score = beam_score.unsqueeze(1)
                repeat_dims = [1] * beam_score.dim()
                repeat_dims[1] = score.size(1)
                beam_score = beam_score.repeat(*repeat_dims)
                score += beam_score
                beam_scores[:, b_idx * decoder.output_size: (b_idx + 1) * decoder.output_size] = score
                all_hidden[:, b_idx * batch_size:(b_idx + 1) * batch_size, :] = decoder_hidden

                beam_list[b_idx].all_output[t] = decoder_output
                all_outputs[:, batch_size * b_idx: batch_size * (b_idx + 1), :] = \
                    beam_list[b_idx].all_output
            topv, topi = beam_scores.topk(beam_size, dim=1)
            beam_list = list()

            for k in range(beam_size):
                temp_topk = topi[:, k]
                temp_input = temp_topk % decoder.output_size
                temp_input = temp_input.data
                if USE_CUDA:
                    temp_input = temp_input.cpu()
                temp_beam_pos = temp_topk / decoder.output_size

                indices = torch.LongTensor(range(batch_size))
                if USE_CUDA:
                    indices = indices.cuda()
                indices += temp_beam_pos * batch_size

                temp_hidden = all_hidden.index_select(1, indices)
                temp_output = all_outputs.index_select(1, indices)

                beam_list.append(Beam(topv[:, k], temp_input, temp_hidden, temp_output))
        all_decoder_outputs = beam_list[0].all_output

        for t in range(max_target_length):
            target[t] = generate_decoder_input(
                target[t], all_decoder_outputs[t], nums_stack_batch, num_start, unk)
    # Loss calculation and backpropagation

    if USE_CUDA:
        target = target.cuda()

    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target.transpose(0, 1).contiguous(),  # -> batch x seq
        target_length
    )

    loss.backward()
    return_loss = loss.item()

    # Clip gradient norms
    if clip:
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return return_loss


def evaluate_attn(input_seq, input_length, num_list, copy_nums, generate_nums, encoder, decoder, output_lang,
                  beam_size=1, english=False, max_length=MAX_OUTPUT_LENGTH):
    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    num_start = output_lang.n_words - copy_nums - 2

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_seq).unsqueeze(1)
    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()

    # Set to not-training mode to disable dropout
    encoder.eval()
    decoder.eval()

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_var, [input_length], None)

    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([output_lang.word2index["SOS"]])  # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
    beam_list = list()
    score = 0
    beam_list.append(Beam(score, decoder_input, decoder_hidden, []))

    # Run through decoder
    for di in range(max_length):
        temp_list = list()
        beam_len = len(beam_list)
        for xb in beam_list:
            if int(xb.input_var[0]) == output_lang.word2index["EOS"]:
                temp_list.append(xb)
                beam_len -= 1
        if beam_len == 0:
            return beam_list[0].all_output
        beam_scores = torch.zeros(decoder.output_size * beam_len)
        hidden_size_0 = decoder_hidden.size(0)
        hidden_size_2 = decoder_hidden.size(2)
        all_hidden = torch.zeros(beam_len, hidden_size_0, 1, hidden_size_2)
        if USE_CUDA:
            beam_scores = beam_scores.cuda()
            all_hidden = all_hidden.cuda()
        all_outputs = []
        current_idx = -1

        for b_idx in range(len(beam_list)):
            decoder_input = beam_list[b_idx].input_var
            if int(decoder_input[0]) == output_lang.word2index["EOS"]:
                continue
            current_idx += 1
            decoder_hidden = beam_list[b_idx].hidden

            # rule_mask = generate_rule_mask(decoder_input, [num_list], output_lang.word2index,
            #                                1, num_start, copy_nums, generate_nums, english)
            if USE_CUDA:
                # rule_mask = rule_mask.cuda()
                decoder_input = decoder_input.cuda()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            # score = f.log_softmax(decoder_output, dim=1) + rule_mask.squeeze()
            score = f.log_softmax(decoder_output, dim=1)
            score += beam_list[b_idx].score
            beam_scores[current_idx * decoder.output_size: (current_idx + 1) * decoder.output_size] = score
            all_hidden[current_idx] = decoder_hidden
            all_outputs.append(beam_list[b_idx].all_output)
        topv, topi = beam_scores.topk(beam_size)

        for k in range(beam_size):
            word_n = int(topi[k])
            word_input = word_n % decoder.output_size
            temp_input = torch.LongTensor([word_input])
            indices = int(word_n / decoder.output_size)

            temp_hidden = all_hidden[indices]
            temp_output = all_outputs[indices]+[word_input]
            temp_list.append(Beam(float(topv[k]), temp_input, temp_hidden, temp_output))

        temp_list = sorted(temp_list, key=lambda x: x.score, reverse=True)

        if len(temp_list) < beam_size:
            beam_list = temp_list
        else:
            beam_list = temp_list[:beam_size]
    return beam_list[0].all_output


def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal


class Tree():
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = []

    def __str__(self, level=0):
        ret = ""
        for child in self.children:
            if isinstance(child, type(self)):
                ret += child.__str__(level + 1)
            else:
                ret += "\t" * level + str(child) + "\n"
        return ret

    def add_child(self, c):
        if isinstance(c, type(self)):
            c.parent = self
        self.children.append(c)
        self.num_children = self.num_children + 1

    def to_string(self):
        r_list = []
        for i in range(self.num_children):
            if isinstance(self.children[i], Tree):
                r_list.append("( " + self.children[i].to_string() + " )")
            else:
                r_list.append(str(self.children[i]))
        return "".join(r_list)

    def flatten(self, output_lang):
        r_list = []
        for i in range(self.num_children):
            if isinstance(self.children[i], type(self)):
                r_list.append(output_lang.word2index["<IS>"])
                cl = self.children[i].flatten(output_lang)
                for k in range(len(cl)):
                    r_list.append(cl[k])
                r_list.append(output_lang.word2index["<IE>"])
            else:
                r_list.append(self.children[i])
        return r_list

    def tree_to_list(self, output_lang):
        list = []
        for i in range(self.num_children):
            if isinstance(self.children[i], type(self)):
                list.append(self.children[i].tree_to_list(output_lang))
            else:
                list.append(output_lang.index2word[self.children[i]])
        return list


def get_dec_batch(dec_tree_batch, batch_size, using_gpu, output_lang):
    queue_tree = {}
    for i in range(1, batch_size + 1):
        queue_tree[i] = []
        queue_tree[i].append({"tree": dec_tree_batch[i - 1],
                              "parent": 0, "child_index": 1})

    cur_index, max_index = 1, 1
    dec_batch = {}
    # max_index: the max number of sequence decoder in one batch
    while (cur_index <= max_index):
        max_w_len = -1
        batch_w_list = []
        for i in range(1, batch_size + 1):
            w_list = []
            if (cur_index <= len(queue_tree[i])):
                t = queue_tree[i][cur_index - 1]["tree"]

                for ic in range(t.num_children):
                    if isinstance(t.children[ic], Tree):
                        # 4가 n?
                        w_list.append(output_lang.word2index['<IE>'])
                        queue_tree[i].append({"tree": t.children[ic],
                                              "parent": cur_index,
                                              "child_index": ic + 1})
                    else:
                        w_list.append(t.children[ic])
                if len(queue_tree[i]) > max_index:
                    max_index = len(queue_tree[i])
            if len(w_list) > max_w_len:
                max_w_len = len(w_list)
            batch_w_list.append(w_list)
        dec_batch[cur_index] = torch.zeros((batch_size,
                                            max_w_len + 2), dtype=torch.long)
        for i in range(batch_size):
            w_list = batch_w_list[i]
            if len(w_list) > 0:
                for j in range(len(w_list)):
                    dec_batch[cur_index][i][j + 1] = w_list[j]
                # add <S>, <E>
                if cur_index == 1:
                    dec_batch[cur_index][i][0] = output_lang.word2index['<S>']
                else:
                    dec_batch[cur_index][i][0] = output_lang.word2index['<IS>']
                dec_batch[cur_index][i][len(w_list) + 1] = output_lang.word2index['<E>']

        if using_gpu:
            dec_batch[cur_index] = dec_batch[cur_index].cuda()
        cur_index += 1

    return dec_batch, queue_tree, max_index

def list_to_tree(r_list, initial=False, depth=0):
   t = Tree()
   if initial:
       t.add_child(r_list[0])
       # print(r_list[0])
       input_len = len(r_list)
       for i in range(1, input_len):
           if isinstance(r_list[i], list):
               t.add_child(list_to_tree(r_list[i], depth=depth + 1))
           else:
               t.add_child(r_list[i])
               # print('\t' * depth + str(r_list[i]))
       return t

   else:
       input_len = len(r_list)
       for i in range(input_len):
           if isinstance(r_list[i], list):
               t.add_child(list_to_tree(r_list[i], depth=depth + 1))
           else:
               t.add_child(r_list[i])
               # print('\t' * depth + str(r_list[i]))
       return t


def recursive_solve(encoder_outputs, graph_embedding, attention_inputs,
                    dec_batch, queue_tree, max_index,
                    dec_seq_length, using_gpu, batch_size, rnn_size,
                    decoder, attention_decoder, mask_batch, reducer, output_lang):

    teacher_force_ratio = 1.0

    criterion = torch.nn.NLLLoss(size_average=False, ignore_index=0)

    loss = 0
    cur_index = 1

    dec_s = {}
    tp_mask_batch = {}
    for i in range(dec_seq_length + 1):
        dec_s[i] = {}
        tp_mask_batch[i] = {}
        for j in range(dec_seq_length + 1):
            dec_s[i][j] = {}
            tp_mask_batch[i][j] = None
    
    for b_id, m_batch in enumerate(mask_batch):
        for m_id, m_list in m_batch.items():
            for w_id, w_list in enumerate(m_list):

                temp_mask = f.one_hot(torch.tensor(w_list), num_classes = output_lang.n_words)

                if len(temp_mask.size()) > 1:
                    temp_mask = temp_mask.sum(dim=0)

                if tp_mask_batch[int(m_id)][w_id] is None:
                    tp_mask_batch[int(m_id)][w_id] = torch.zeros(len(mask_batch), output_lang.n_words)

                tp_mask_batch[int(m_id)][w_id][b_id] = temp_mask

    # graph_cell_state = torch.zeros((opt.batch_size, opt.rnn_size), dtype=torch.float, requires_grad=True)
    # graph_hidden_state = torch.zeros((opt.batch_size, opt.rnn_size), dtype=torch.float, requires_grad=True)
    # if opt.using_gpu:
    #     graph_cell_state = graph_cell_state.cuda()
    #     graph_hidden_state = graph_hidden_state.cuda()

    # graph_embedding, _ = torch.min(encoder_outputs, 0)
    # graph_embedding, _ = torch.max(encoder_outputs, 0)
    # graph_embedding = torch.mean(encoder_outputs, 0)
    graph_cell_state = graph_embedding
    graph_hidden_state = graph_embedding

    encoder_outputs = encoder_outputs.transpose(0, 1)
    # bigru_outputs = bigru_outputs.transpose(0, 1)
    structural_info = encoder_outputs

    while (cur_index <= max_index):
        for j in range(1, 3):
            # dec_s[cur_index][0][j] = torch.zeros((batch_size, rnn_size), dtype=torch.float, requires_grad=True)
            dec_s[cur_index][0][j] = torch.zeros((batch_size, rnn_size), dtype=torch.float, requires_grad=True)
            if using_gpu:
                dec_s[cur_index][0][j] = dec_s[cur_index][0][j].cuda()

        sibling_state = torch.zeros((batch_size, rnn_size), dtype=torch.float, requires_grad=True)
        if using_gpu:
            sibling_state = sibling_state.cuda()

        if cur_index == 1:
            for i in range(batch_size):
                dec_s[1][0][1][i, :] = graph_cell_state[i]
                dec_s[1][0][2][i, :] = graph_hidden_state[i]
            # dec_s[1][0][1] = graph_cell_state
            # dec_s[1][0][2] = graph_hidden_state

        else:
            for i in range(1, batch_size + 1):
                if (cur_index <= len(queue_tree[i])):
                    par_index = queue_tree[i][cur_index - 1]["parent"]
                    child_index = queue_tree[i][cur_index - 1]["child_index"]

                    dec_s[cur_index][0][1][i - 1, :] = \
                        dec_s[par_index][child_index][1][i - 1, :]
                    dec_s[cur_index][0][2][i - 1, :] = dec_s[par_index][child_index][2][i - 1, :]

                flag_sibling = False
                for q_index in range(len(queue_tree[i])):
                    if (cur_index <= len(queue_tree[i])) and (q_index < cur_index - 1) and \
                            (queue_tree[i][q_index]["parent"] == queue_tree[i][cur_index - 1]["parent"]) and \
                            (queue_tree[i][q_index]["child_index"] < queue_tree[i][cur_index - 1]["child_index"]):
                        flag_sibling = True
                        sibling_index = q_index
                if flag_sibling:
                    sibling_state[i - 1, :] = dec_s[sibling_index][dec_batch[sibling_index].size(1) - 1][2][i - 1, :]

        parent_h = dec_s[cur_index][0][2]
        if using_gpu:
            parent_h.cuda()
        for i in range(dec_batch[cur_index].size(1) - 1):
            teacher_force = random.random() < teacher_force_ratio
            if teacher_force != True and i > 0:
                input_word = pred.argmax(1)
            else:
                input_word = dec_batch[cur_index][:, i]
                if using_gpu:
                    input_word = input_word.cuda()

            dec_s[cur_index][i + 1][1], dec_s[cur_index][i + 1][2] = decoder(input_word, dec_s[cur_index][i][1],
                                                                             dec_s[cur_index][i][2], parent_h,
                                                                             sibling_state)
            # structural_info -> Bi-LSTM

            mask = tp_mask_batch[cur_index][i]
            pred = attention_decoder(attention_inputs[0], dec_s[cur_index][i + 1][2], attention_inputs[1], mask)
            gt = dec_batch[cur_index][:, i + 1]
            if using_gpu:
                gt = gt.cuda()
            loss += criterion(pred, gt)
        cur_index = cur_index + 1

    return loss

def train_tree(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, num_value_batch, generate_nums,
               embedding, encoder, decoder, attention_decoder, embedding_optimizer, encoder_optimizer, decoder_optimizer, attention_decoder_optimizer,
               input_lang, output_lang, num_pos, batch_graph, contextual_input, dec_batch, queue_tree, max_index, mask_batch, reducer):
    # sequence mask for attention
    # seq_mask = []
    # max_len = max(input_length)
    # for i in input_length:
    #     seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    # seq_mask = torch.ByteTensor(seq_mask)

    # num_mask = []
    # max_num_size = max(num_size_batch) + len(generate_nums)
    # for i in num_size_batch:
    #     d = i + len(generate_nums)
    #     num_mask.append([0] * d + [1] * (max_num_size - d))
    # num_mask = torch.ByteTensor(num_mask)

    # unk = output_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    # input_var = torch.LongTensor(input_batch).transpose(0, 1)

    # target = torch.LongTensor(target_batch).transpose(0, 1)
    # batch_graph = torch.LongTensor(batch_graph)

    # padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    embedding.train()
    encoder.train()
    decoder.train()
    attention_decoder.train()

    # if USE_CUDA:
    #     input_var = input_var.cuda()
    #     # seq_mask = seq_mask.cuda()
    #     # padding_hidden = padding_hidden.cuda()
    #     # num_mask = num_mask.cuda()
    #     batch_graph = batch_graph.cuda()

    # Zero gradients of both optimizers
    embedding_optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    attention_decoder_optimizer.zero_grad()

    # ===========================================for BERT===================================================
    # orig_idx = None
    # embedded = None
    # if config.embedding == 'bert' or config.embedding == 'roberta':
    if True:
        # contextual_input = index_batch_to_words(input_batch, input_length, input_lang)

        input_seq1, input_len1, token_ids, index_retrieve = embedding(contextual_input)
        num_pos = index_retrieve.copy()

        new_group_batch = allocate_group_num(index_retrieve, input_len1)
        batch_graph = get_single_batch_graph(token_ids.cpu().tolist(), input_len1, new_group_batch, num_value_batch,
                                             num_pos)
        batch_graph = torch.LongTensor(batch_graph)

        # print(num_value_batch, num_pos)

        input_seq1 = input_seq1.transpose(0, 1)
        # embedded, input_length, orig_idx = sort_by_len(input_seq1, input_len1, "cuda:0")
        embedded, input_length, orig_idx = sort_by_len(input_seq1, input_len1, "cuda" if USE_CUDA else "cpu")
        # print(embedded.size())
        input_length = torch.IntTensor(input_length)

    if USE_CUDA:
        batch_graph = batch_graph.cuda()

    encoder_outputs, problem_output, graph_embedding, attention_inputs = \
        encoder(embedded, input_length, orig_idx, batch_graph)

    # ===============changed=================
    # sequence mask for attention
    # seq_mask = []
    # max_len = max(input_length)
    # for i in input_length:
    #     seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    # seq_mask = torch.BoolTensor(seq_mask)
    #
    # if USE_CUDA:
    #     seq_mask = seq_mask.cuda()
    # ==============================================================================================


    # Run words through encoder
    # encoder_outputs, problem_output, bigru_outputs = encoder(input_var, input_length, batch_graph)
    # Prepare input and output variables
    # node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
    #
    # max_target_length = max(target_length)
    #
    # all_node_outputs = []
    # all_leafs = []
    #
    # copy_num_len = [len(_) for _ in num_pos]
    # num_size = max(copy_num_len)
    # all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
    #                                                           encoder.hidden_size)

    # num_start = output_lang.num_start
    # embeddings_stacks = [[] for _ in range(batch_size)]
    # left_childs = [None for _ in range(batch_size)]
    # for t in range(max_target_length):
    #     num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
    #         node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
    #
    #     # all_leafs.append(p_leaf)
    #     outputs = torch.cat((op, num_score), 1)
    #     all_node_outputs.append(outputs)
    #
    #     target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
    #     target[t] = target_t
    #     if USE_CUDA:
    #         generate_input = generate_input.cuda()
    #     left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
    #     left_childs = []
    #     for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
    #                                            node_stacks, target[t].tolist(), embeddings_stacks):
    #         if len(node_stack) != 0:
    #             node = node_stack.pop()
    #         else:
    #             left_childs.append(None)
    #             continue
    #
    #         if i < num_start:
    #             node_stack.append(TreeNode(r))
    #             node_stack.append(TreeNode(l, left_flag=True))
    #             o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
    #         else:
    #             current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
    #             while len(o) > 0 and o[-1].terminal:
    #                 sub_stree = o.pop()
    #                 op = o.pop()
    #                 current_num = merge(op.embedding, sub_stree.embedding, current_num)
    #             o.append(TreeEmbedding(current_num, True))
    #         if len(o) > 0 and o[-1].terminal:
    #             left_childs.append(o[-1].embedding)
    #         else:
    #             left_childs.append(None)

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    # all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    # target = target.transpose(0, 1).contiguous()
    # if USE_CUDA:
    #     # all_leafs = all_leafs.cuda()
    #     all_node_outputs = all_node_outputs.cuda()
    #     target = target.cuda()
    #
    # # op_target = target < num_start
    # # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    # loss = masked_cross_entropy(all_node_outputs, target, target_length)
    # # loss = loss_0 + loss_1
    # loss.backward()
    # # clip the grad
    # loss = loss / opt.batch_size
    # loss.backward()
    # torch.nn.utils.clip_grad_value_(encoder.parameters(), opt.grad_clip)
    # torch.nn.utils.clip_grad_value_(decoder.parameters(), opt.grad_clip)
    # torch.nn.utils.clip_grad_value_(attention_decoder.parameters(), opt.grad_clip)
    # encoder_optimizer.step()
    # decoder_optimizer.step()
    # attention_decoder_optimizer.step()
    #
    # # Update parameters with optimizers
    # encoder_optimizer.step()
    # predict_optimizer.step()
    # generate_optimizer.step()
    # merge_optimizer.step()

    # target_batch = [list_to_tree(l) for l in target_batch]
    #
    # dec_batch, queue_tree, max_index = get_dec_batch(target_batch, batch_size, USE_CUDA, output_lang)

    loss = \
        recursive_solve(encoder_outputs, graph_embedding, attention_inputs,
                        dec_batch, queue_tree, max_index,
                        MAX_OUTPUT_LENGTH, USE_CUDA, batch_size, encoder.hidden_size,
                        decoder, attention_decoder, mask_batch, reducer, output_lang)

    loss = loss / batch_size
    loss.backward()
    torch.nn.utils.clip_grad_value_(embedding.parameters(), 5)
    torch.nn.utils.clip_grad_value_(encoder.parameters(), 5)
    torch.nn.utils.clip_grad_value_(decoder.parameters(), 5)
    torch.nn.utils.clip_grad_value_(attention_decoder.parameters(), 5)

    embedding_optimizer.step()
    encoder_optimizer.step()
    decoder_optimizer.step()
    attention_decoder_optimizer.step()

    return loss


def val_tree(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, num_value_batch, generate_nums,
               embedding, encoder, decoder, attention_decoder, embedding_optimizer, encoder_optimizer, decoder_optimizer, attention_decoder_optimizer,
               input_lang, output_lang, num_pos, batch_graph, contextual_input, dec_batch, queue_tree, max_index, mask_batch, reducer):
    # sequence mask for attention
    # seq_mask = []
    # max_len = max(input_length)
    # for i in input_length:
    #     seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    # seq_mask = torch.ByteTensor(seq_mask)

    # num_mask = []
    # max_num_size = max(num_size_batch) + len(generate_nums)
    # for i in num_size_batch:
    #     d = i + len(generate_nums)
    #     num_mask.append([0] * d + [1] * (max_num_size - d))
    # num_mask = torch.ByteTensor(num_mask)

    # unk = output_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    # input_var = torch.LongTensor(input_batch).transpose(0, 1)

    # target = torch.LongTensor(target_batch).transpose(0, 1)
    # batch_graph = torch.LongTensor(batch_graph)

    # padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    embedding.eval()
    encoder.eval()
    decoder.eval()
    attention_decoder.eval()

    # orig_idx = None
    # embedded = None
    # if config.embedding == 'bert' or config.embedding == 'roberta':
    if True:
        # contextual_input = index_batch_to_words(input_batch, input_length, input_lang)
        input_seq1, input_len1, token_ids, index_retrieve = embedding(contextual_input)
        num_pos = index_retrieve.copy()

        new_group_batch = allocate_group_num(index_retrieve, input_len1)
        # new_group_batch = []
        # for bat in range(len(group_batch)):
        # 	try:
        # 		new_group_batch.append([index_retrieve[bat][index1] for index1 in group_batch[bat] if index1 < len(index_retrieve[bat])])
        # 	except:
        # 		pdb.set_trace()

        batch_graph = get_single_batch_graph(token_ids.cpu().tolist(), input_len1, new_group_batch, num_value_batch,
                                             num_pos)
        batch_graph = torch.LongTensor(batch_graph)

        # print(num_value_batch, num_pos)

        input_seq1 = input_seq1.transpose(0, 1)
        embedded, input_length, orig_idx = sort_by_len(input_seq1, input_len1, "cuda" if USE_CUDA else "cpu")
        # print(embedded.size())
        input_length = torch.IntTensor(input_length)

    if USE_CUDA:
        batch_graph = batch_graph.cuda()

    encoder_outputs, problem_output, graph_embedding, attention_inputs = \
        encoder(embedded, input_length, orig_idx, batch_graph)


    loss = \
        recursive_solve(encoder_outputs, graph_embedding, attention_inputs,
                        dec_batch, queue_tree, max_index,
                        MAX_OUTPUT_LENGTH, USE_CUDA, batch_size, encoder.hidden_size,
                        decoder, attention_decoder, mask_batch, reducer, output_lang)

    loss = loss / batch_size

    return loss

def extract_node_mask_eval(output_lang, parent_word_list, parent_i_child_list, i_child):
    parent_gt = parent_word_list[parent_i_child_list[i_child-1]]
    child_diff = 0
    for pp in reversed(parent_word_list[:parent_i_child_list[i_child-1]]):
        # if pp != output_lang.get_symbol_idx('<N>'):
        if pp != output_lang.word2index('<IE>'):
            break
        child_diff += 1
    child_idx = child_diff

    return parent_gt, child_idx

def evaluate_tree(input_batch, input_length, embedding, encoder, decoder, attention_decoder, reducer,
                  input_lang, output_lang, num_value, num_pos=None, batch_graph=None, beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):
    #
    # # seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    # input_var = torch.LongTensor(input_batch).unsqueeze(1)
    # # batch_graph = torch.LongTensor(batch_graph)
    #
    # # num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)).fill_(0)
    #
    # # Set to not-training mode to disable dropout
    # embedding.eval()
    # encoder.eval()
    # decoder.eval()
    # attention_decoder.eval()
    #
    # # padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    #
    # batch_size = 1
    #
    # if USE_CUDA:
    #     input_var = input_var.cuda()
    #     # seq_mask = seq_mask.cuda()
    #     # padding_hidden = padding_hidden.cuda()
    #     # num_mask = num_mask.cuda()
    #     # batch_graph = batch_graph.cuda()
    #
    # orig_idx = None
    # embedded = None
    # # if config.embedding == 'bert' or config.embedding == 'roberta':
    # if True:
    #     contextual_input = index_batch_to_words([input_batch], [input_length], input_lang)
    #     input_seq1, input_len1, token_ids, index_retrieve = embedding(contextual_input)
    #     num_pos = index_retrieve.copy()[0]
    #
    #     new_group_batch = allocate_group_num(index_retrieve, input_len1)[0]
    #     # print(new_group_batch, token_ids.cpu().tolist()[0],  input_len1[0], num_value, num_pos)
    #     # new_group_batch = []
    #     # for bat in range(len(group_batch)):
    #     # 	try:
    #     # 		new_group_batch.append([index_retrieve[bat][index1] for index1 in group_batch[bat] if index1 < len(index_retrieve[bat])])
    #     # 	except:
    #     # 		pdb.set_trace()
    #
    #     batch_graph = get_single_example_graph(token_ids.cpu().tolist()[0], input_len1[0], new_group_batch, num_value,
    #                                          num_pos)
    #     batch_graph = torch.LongTensor(batch_graph)
    #
    #     input_seq1 = input_seq1.transpose(0, 1)
    #     embedded, input_length, orig_idx = sort_by_len(input_seq1, input_len1, "cuda" if USE_CUDA else "cpu")
    #     # input_length = input_length[0]
    #     input_length = torch.IntTensor(input_length)
    #
    # if USE_CUDA:
    #     batch_graph = batch_graph.cuda()
    #
    # # Run words through encoder
    # encoder_outputs, problem_output, graph_embedding, attention_inputs = \
    #     encoder(embedded, input_length, orig_idx, batch_graph)
    #
    # encoder_outputs = encoder_outputs.transpose(0, 1)
    # prev_c = graph_embedding
    # prev_h = graph_embedding
    #
    # queue_decode = []
    # queue_decode.append({"s": (prev_c, prev_h), "parent": 0, "child_index": 1, "t": Tree()})
    # head = 1
    # while head <= len(queue_decode) and head <= MAX_OUTPUT_LENGTH:
    #     s = queue_decode[head - 1]["s"]
    #     parent_h = s[1]
    #     t = queue_decode[head - 1]["t"]
    #
    #     sibling_state = torch.zeros((1, encoder.hidden_size), dtype=torch.float, requires_grad=False)
    #
    #     if USE_CUDA:
    #         sibling_state = sibling_state.cuda()
    #     flag_sibling = False
    #     for q_index in range(len(queue_decode)):
    #         if (head <= len(queue_decode)) and (q_index < head - 1) and (
    #                 queue_decode[q_index]["parent"] == queue_decode[head - 1]["parent"]) and (
    #                 queue_decode[q_index]["child_index"] < queue_decode[head - 1]["child_index"]):
    #             flag_sibling = True
    #             sibling_index = q_index
    #     if flag_sibling:
    #         sibling_state = queue_decode[sibling_index]["s"][1]
    #
    #     if head == 1:
    #         prev_word = torch.tensor([output_lang.word2index['<S>']], dtype=torch.long)
    #     else:
    #         prev_word = torch.tensor([output_lang.word2index['<IS>']], dtype=torch.long)
    #
    #     if USE_CUDA:
    #         prev_word = prev_word.cuda()
    #
    #     if head != 1:
    #         parent_word_list = queue_decode[queue_decode[head - 1]["parent"]-1]['t'].children
    #         child_idx = queue_decode[head - 1]["child_index"]-1
    #         parent_gt = parent_word_list[:child_idx+2][::-1][child_idx+1]
    #     else:
    #         parent_gt = None
    #         child_idx = None
    #
    #     i_child = 0
    #     prev_word_list = []
    #     prev_word_list.append(prev_word.item())
    #
    #     while True:
    #         mask = reducer.reduce_out([parent_gt], [child_idx], [prev_word_list])
    #         one_hot_mask = f.one_hot(torch.tensor(mask[0]), num_classes=output_lang.n_words).sum(dim=0).unsqueeze(0)
    #         curr_c, curr_h = decoder(prev_word, s[0], s[1], parent_h, sibling_state)
    #         #mask 생성 타이밍
    #         one_hot_mask=None
    #         prediction = attention_decoder(attention_inputs[0], curr_h, attention_inputs[1], one_hot_mask)
    #
    #         s = (curr_c, curr_h)
    #         _, _prev_word = prediction.max(1)
    #         prev_word = _prev_word
    #
    #         if int(prev_word[0]) == output_lang.word2index['<E>'] or \
    #                 t.num_children >= max_length:
    #             break
    #         elif int(prev_word[0]) == output_lang.word2index['<IE>']:
    #             queue_decode.append(
    #                 {"s": (s[0].clone(), s[1].clone()), "parent": head, "child_index": i_child, "t": Tree()})
    #             t.add_child(int(prev_word[0]))
    #         else:
    #             t.add_child(int(prev_word[0]))
    #         i_child = i_child + 1
    #         prev_word_list.append(prev_word.item())
    #     head = head + 1
    # for i in range(len(queue_decode) - 1, 0, -1):
    #     cur = queue_decode[i]
    #     queue_decode[cur["parent"] - 1]["t"].children[cur["child_index"] - 1] = cur["t"]

    # seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)
    # batch_graph = torch.LongTensor(batch_graph)

    # num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    embedding.eval()
    encoder.eval()
    decoder.eval()
    attention_decoder.eval()

    # padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        # seq_mask = seq_mask.cuda()
        # padding_hidden = padding_hidden.cuda()
        # num_mask = num_mask.cuda()
        # batch_graph = batch_graph.cuda()

    orig_idx = None
    embedded = None
    # if config.embedding == 'bert' or config.embedding == 'roberta':
    if True:
        contextual_input = index_batch_to_words([input_batch], [input_length], input_lang)
        input_seq1, input_len1, token_ids, index_retrieve = embedding(contextual_input)
        num_pos = index_retrieve.copy()[0]

        new_group_batch = allocate_group_num(index_retrieve, input_len1)[0]
        # print(new_group_batch, token_ids.cpu().tolist()[0],  input_len1[0], num_value, num_pos)
        # new_group_batch = []
        # for bat in range(len(group_batch)):
        # 	try:
        # 		new_group_batch.append([index_retrieve[bat][index1] for index1 in group_batch[bat] if index1 < len(index_retrieve[bat])])
        # 	except:
        # 		pdb.set_trace()

        batch_graph = get_single_example_graph(token_ids.cpu().tolist()[0], input_len1[0], new_group_batch, num_value,
                                               num_pos)
        batch_graph = torch.LongTensor(batch_graph)

        input_seq1 = input_seq1.transpose(0, 1)
        embedded, input_length, orig_idx = sort_by_len(input_seq1, input_len1, "cuda" if USE_CUDA else "cpu")
        # input_length = input_length[0]
        input_length = torch.IntTensor(input_length)

    if USE_CUDA:
        batch_graph = batch_graph.cuda()

    # Run words through encoder
    encoder_outputs, problem_output, graph_embedding, attention_inputs = \
        encoder(embedded, input_length, orig_idx, batch_graph)

    encoder_outputs = encoder_outputs.transpose(0, 1)
    prev_c = graph_embedding
    prev_h = graph_embedding

    queue_decode = []
    queue_decode.append({"s": (prev_c, prev_h), "parent": 0, "child_index": 1, "t": Tree()})
    head = 1
    while head <= len(queue_decode) and head <= 100:
        s = queue_decode[head - 1]["s"]
        parent_h = s[1]
        t = queue_decode[head - 1]["t"]

        sibling_state = torch.zeros((1, encoder.hidden_size), dtype=torch.float, requires_grad=False)

        if USE_CUDA:
            sibling_state = sibling_state.cuda()
        flag_sibling = False
        for q_index in range(len(queue_decode)):
            if (head <= len(queue_decode)) and (q_index < head - 1) and (
                    queue_decode[q_index]["parent"] == queue_decode[head - 1]["parent"]) and (
                    queue_decode[q_index]["child_index"] < queue_decode[head - 1]["child_index"]):
                flag_sibling = True
                sibling_index = q_index
        if flag_sibling:
            sibling_state = queue_decode[sibling_index]["s"][1]

        if head == 1:
            prev_word = torch.tensor([output_lang.word2index['<S>']], dtype=torch.long)
        else:
            prev_word = torch.tensor([output_lang.word2index['<IS>']], dtype=torch.long)
        if USE_CUDA:
            prev_word = prev_word.cuda()
        i_child = 1
        while True:
            curr_c, curr_h = decoder(prev_word, s[0], s[1], parent_h, sibling_state)
            prediction = attention_decoder(attention_inputs[0], curr_h, attention_inputs[1])

            s = (curr_c, curr_h)
            _, _prev_word = prediction.max(1)
            prev_word = _prev_word

            if int(prev_word[0]) == output_lang.word2index['<E>'] or \
                    t.num_children >= max_length:
                break
            elif int(prev_word[0]) == output_lang.word2index['<IE>']:
                queue_decode.append(
                    {"s": (s[0].clone(), s[1].clone()), "parent": head, "child_index": i_child, "t": Tree()})
                t.add_child(int(prev_word[0]))
            else:
                t.add_child(int(prev_word[0]))
            i_child = i_child + 1
        head = head + 1
    for i in range(len(queue_decode) - 1, 0, -1):
        cur = queue_decode[i]
        queue_decode[cur["parent"] - 1]["t"].children[cur["child_index"] - 1] = cur["t"]

    return queue_decode[0]["t"].tree_to_list(output_lang)
    # return queue_decode[0]["t"]
    # return queue_decode

def evaluate_tree_ensemble(input_batch, input_length, generate_nums, embeddings, encoders, decoders, attention_decoders,
                           input_lang, output_lang, num_value, num_pos, batch_graph, beam_size=5, english=False,
                           max_length=MAX_OUTPUT_LENGTH):

    # seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)
    batch_graph = torch.LongTensor(batch_graph)

    # num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    num_models = len(attention_decoders)
    for model_i in range(num_models):
        embeddings[model_i].eval()
        encoders[model_i].eval()
        decoders[model_i].eval()
        attention_decoders[model_i].eval()
    # padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        # seq_mask = seq_mask.cuda()
        # padding_hidden = padding_hidden.cuda()
        # num_mask = num_mask.cuda()
        batch_graph = batch_graph.cuda()

    if USE_CUDA:
        batch_graph = batch_graph.cuda()

    seq_mask = torch.BoolTensor(1, input_length).fill_(0)

    if USE_CUDA:
        seq_mask = seq_mask.cuda()

    # Run words through encoder
    # encoder_outputs, problem_output, bigru_outputs = encoder(input_var, [input_length], batch_graph)

    # Prepare input and output variables
    # node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    # num_size = len(num_pos)
    # all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
    #                                                           encoder.hidden_size)
    # num_start = output_lang.num_start
    # # B x P x N
    # embeddings_stacks = [[] for _ in range(batch_size)]
    # left_childs = [None for _ in range(batch_size)]
    #
    # beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]
    #
    # for t in range(max_length):
    #     current_beams = []
    #     while len(beams) > 0:
    #         b = beams.pop()
    #         if len(b.node_stack[0]) == 0:
    #             current_beams.append(b)
    #             continue
    #         # left_childs = torch.stack(b.left_childs)
    #         left_childs = b.left_childs
    #
    #         num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
    #             b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
    #             seq_mask, num_mask)
    #
    #         # leaf = p_leaf[:, 0].unsqueeze(1)
    #         # repeat_dims = [1] * leaf.dim()
    #         # repeat_dims[1] = op.size(1)
    #         # leaf = leaf.repeat(*repeat_dims)
    #         #
    #         # non_leaf = p_leaf[:, 1].unsqueeze(1)
    #         # repeat_dims = [1] * non_leaf.dim()
    #         # repeat_dims[1] = num_score.size(1)
    #         # non_leaf = non_leaf.repeat(*repeat_dims)
    #         #
    #         # p_leaf = torch.cat((leaf, non_leaf), dim=1)
    #         out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)
    #
    #         # out_score = p_leaf * out_score
    #
    #         topv, topi = out_score.topk(beam_size)
    #
    #         # is_leaf = int(topi[0])
    #         # if is_leaf:
    #         #     topv, topi = op.topk(1)
    #         #     out_token = int(topi[0])
    #         # else:
    #         #     topv, topi = num_score.topk(1)
    #         #     out_token = int(topi[0]) + num_start
    #
    #         for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
    #             current_node_stack = copy_list(b.node_stack)
    #             current_left_childs = []
    #             current_embeddings_stacks = copy_list(b.embedding_stack)
    #             current_out = copy.deepcopy(b.out)
    #
    #             out_token = int(ti)
    #             current_out.append(out_token)
    #
    #             node = current_node_stack[0].pop()
    #
    #             if out_token < num_start:
    #                 generate_input = torch.LongTensor([out_token])
    #                 if USE_CUDA:
    #                     generate_input = generate_input.cuda()
    #                 left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
    #
    #                 current_node_stack[0].append(TreeNode(right_child))
    #                 current_node_stack[0].append(TreeNode(left_child, left_flag=True))
    #
    #                 current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
    #             else:
    #                 current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)
    #
    #                 while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
    #                     sub_stree = current_embeddings_stacks[0].pop()
    #                     op = current_embeddings_stacks[0].pop()
    #                     current_num = merge(op.embedding, sub_stree.embedding, current_num)
    #                 current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
    #             if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
    #                 current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
    #             else:
    #                 current_left_childs.append(None)
    #             current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
    #                                           current_left_childs, current_out))
    #     beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
    #     beams = beams[:beam_size]
    #     flag = True
    #     for b in beams:
    #         if len(b.node_stack[0]) != 0:
    #             flag = False
    #     if flag:
    #         break

    all_encoder_outputs = list()
    for model_i in range(num_models):
        orig_idx = None
        embedded = None
        # if config.embedding == 'bert' or config.embedding == 'roberta':
        if True:
            contextual_input = index_batch_to_words([input_batch], [input_length], input_lang)
            input_seq1, input_len1, token_ids, index_retrieve = embeddings[model_i](contextual_input)
            num_pos = index_retrieve.copy()[0]

            new_group_batch = allocate_group_num(index_retrieve, input_len1)[0]
            # print(new_group_batch, token_ids.cpu().tolist()[0],  input_len1[0], num_value, num_pos)
            # new_group_batch = []
            # for bat in range(len(group_batch)):
            # 	try:
            # 		new_group_batch.append([index_retrieve[bat][index1] for index1 in group_batch[bat] if index1 < len(index_retrieve[bat])])
            # 	except:
            # 		pdb.set_trace()

            batch_graph = get_single_example_graph(token_ids.cpu().tolist()[0], input_len1[0], new_group_batch,
                                                   num_value,
                                                   num_pos)
            batch_graph = torch.LongTensor(batch_graph)

            input_seq1 = input_seq1.transpose(0, 1)
            embedded, input_length, orig_idx = sort_by_len(input_seq1, input_len1, "cuda" if USE_CUDA else "cpu")
            # input_length = input_length[0]
            input_length = torch.IntTensor(input_length)

        encoder_outputs, problem_output, graph_embedding, attention_inputs = \
            encoders[model_i](embedded, input_var, input_length, batch_graph)
        all_encoder_outputs.append((encoder_outputs.transpose(0, 1), graph_embedding, attention_inputs))

    s = [(out[1], out[1]) for out in all_encoder_outputs]

    queue_decode = []
    queue_decode.append({"s": s, "parent": 0, "child_index": 1, "t": Tree()})
    head = 1
    while head <= len(queue_decode) and head <= max_length:
        s = queue_decode[head - 1]["s"]
        parent_h = [ss[1] for ss in s]
        t = queue_decode[head - 1]["t"]

        sibling_state = [torch.zeros((1, encoders[0].hidden_size), dtype=torch.float, requires_grad=False)
                         for _ in range(num_models)]

        if USE_CUDA:
            sibling_state = [s.cuda() for s in sibling_state]
        flag_sibling = False
        for q_index in range(len(queue_decode)):
            if (head <= len(queue_decode)) and (q_index < head - 1) and (
                    queue_decode[q_index]["parent"] == queue_decode[head - 1]["parent"]) and (
                    queue_decode[q_index]["child_index"] < queue_decode[head - 1]["child_index"]):
                flag_sibling = True
                sibling_index = q_index
        if flag_sibling:
            sibling_state = queue_decode[sibling_index]["s"][1]

        if head == 1:
            prev_word = torch.tensor([output_lang.word2index['<S>']], dtype=torch.long)
        else:
            prev_word = torch.tensor([output_lang.word2index['<IS>']], dtype=torch.long)
        if USE_CUDA:
            prev_word = prev_word.cuda()
        i_child = 1
        while True:
            cur_s = list()
            predictions = list()
            for model_i in range(num_models):
                curr_c, curr_h = decoders[model_i](prev_word, s[model_i][0], s[model_i][1],
                                                   parent_h[model_i], sibling_state[model_i])
                cur_s.append((curr_c, curr_h))
                attention_inputs = all_encoder_outputs[model_i][2]
                prediction = attention_decoders[model_i](attention_inputs[0], curr_h, attention_inputs[1])
                predictions.append(nn.functional.softmax(prediction, dim=1))
            prediction = torch.mean(torch.stack(predictions, dim=0), dim=0)

            s = cur_s
            # s = (curr_c, curr_h)
            _, _prev_word = prediction.max(1)
            prev_word = _prev_word

            if int(prev_word[0]) == output_lang.word2index['<E>'] or t.num_children >= max_length:
                break
            elif int(prev_word[0]) == output_lang.word2index['<IE>']:
                queue_decode.append(
                    {"s": [(ss[0].clone(), ss[1].clone()) for ss in s],
                     "parent": head, "child_index": i_child, "t": Tree()})
                t.add_child(int(prev_word[0]))
            else:
                t.add_child(int(prev_word[0]))
            i_child = i_child + 1
        head = head + 1
    for i in range(len(queue_decode) - 1, 0, -1):
        cur = queue_decode[i]
        queue_decode[cur["parent"] - 1]["t"].children[cur["child_index"] - 1] = cur["t"]

    return queue_decode[0]["t"].flatten(output_lang)

def beam_copy(beam):
    # beams = [{"q": list([{"s": s, "parent": 0, "child_index": 1, "t": Tree()}]),
    #           "score": 0.0, "score_length": 0.0,
    #           "head": 1, "child": 1, "head_done": False}]
    new_beam = dict(beam)
    q = beam["q"]
    new_q = list()
    for qq in q:
        # {"s": s, "parent": 0, "child_index": 1, "t": Tree()}
        new_q.append({"s": [(qq_s[0].clone(), qq_s[1].clone()) for qq_s in qq["s"]],
                      "parent": qq["parent"], "child_index": qq["child_index"],
                      "t": copy.deepcopy(qq["t"])})
    new_beam["q"] = new_q
    new_beam["parent_h"] = [p_h.clone() for p_h in beam["parent_h"]]
    new_beam["prev_word"] = beam["prev_word"].clone()
    new_beam["sibling_state"] = [s.clone() for s in beam["sibling_state"]]

    return new_beam

def evaluate_tree_ensemble_beam_search(input_batch, input_length, generate_nums,
                                       embeddings, encoders, decoders, attention_decoders,
                                       input_lang, output_lang, num_value, beam_size=5,
                                       max_length=MAX_OUTPUT_LENGTH):
    input_var = torch.LongTensor(input_batch).unsqueeze(1)

    # Set to not-training mode to disable dropout
    num_models = len(attention_decoders)
    for model_i in range(num_models):
        embeddings[model_i].eval()
        encoders[model_i].eval()
        decoders[model_i].eval()
        attention_decoders[model_i].eval()

    if USE_CUDA:
        input_var = input_var.cuda()

    all_encoder_outputs = list()
    for model_i in range(num_models):
        contextual_input = index_batch_to_words([input_batch], [input_length], input_lang)
        input_seq1, input_len1, token_ids, index_retrieve = embeddings[model_i](contextual_input)
        num_pos = index_retrieve.copy()[0]

        new_group_batch = allocate_group_num(index_retrieve, input_len1)[0]
        # print(new_group_batch, token_ids.cpu().tolist()[0],  input_len1[0], num_value, num_pos)
        # new_group_batch = []
        # for bat in range(len(group_batch)):
        # 	try:
        # 		new_group_batch.append([index_retrieve[bat][index1] for index1 in group_batch[bat] if index1 < len(index_retrieve[bat])])
        # 	except:
        # 		pdb.set_trace()

        batch_graph = get_single_example_graph(token_ids.cpu().tolist()[0], input_len1[0], new_group_batch, num_value,
                                               num_pos)
        batch_graph = torch.LongTensor(batch_graph)

        input_seq1 = input_seq1.transpose(0, 1)
        embedded, input_length, orig_idx = sort_by_len(input_seq1, input_len1, "cuda" if USE_CUDA else "cpu")
        # input_length = input_length[0]
        input_length = torch.IntTensor(input_length)

        if USE_CUDA:
            batch_graph = batch_graph.cuda()

        encoder_outputs, problem_output, graph_embedding, attention_inputs = \
            encoders[model_i](embedded, input_length, orig_idx, batch_graph)

        all_encoder_outputs.append((encoder_outputs.transpose(0, 1), graph_embedding, attention_inputs))

    s = [(out[1].clone(), out[1].clone()) for out in all_encoder_outputs]

    beams = [{"q": list([{"s": s, "parent": 0, "child_index": 1, "t": Tree()}]),
              "score": 0.0, "score_length": 0.0,
              "head": 1, "child": 1, "head_done": False}]
    # depth level
    while False in [b["head_done"] for b in beams]:
        # while head <= len(queue_decode) and head <= max_length:
        new_beams = list()
        for b in beams:
            if not b["head_done"]:
                head = b["head"]
                i_child = b["child"]
                queue_decode = b["q"]
                s = queue_decode[head - 1]["s"]

                if i_child == 1:
                    sibling_state = [torch.zeros((1, encoders[0].hidden_size), dtype=torch.float, requires_grad=False)
                                     for _ in range(num_models)]

                    if USE_CUDA:
                        sibling_state = [s.cuda() for s in sibling_state]
                    flag_sibling = False
                    for q_index in range(len(queue_decode)):
                        if (head <= len(queue_decode)) and (q_index < head - 1) and (
                                queue_decode[q_index]["parent"] == queue_decode[head - 1]["parent"]) and (
                                queue_decode[q_index]["child_index"] < queue_decode[head - 1]["child_index"]):
                            flag_sibling = True
                            sibling_index = q_index
                    if flag_sibling:
                        sibling_state = [s[1] for s in queue_decode[sibling_index]["s"]]

                    if head == 1:
                        prev_word = torch.tensor([output_lang.word2index['<S>']], dtype=torch.long)
                    else:
                        prev_word = torch.tensor([output_lang.word2index['<IS>']], dtype=torch.long)
                    if USE_CUDA:
                        prev_word = prev_word.cuda()

                    parent_h = [ss[1] for ss in s]
                else:
                    sibling_state = b["sibling_state"]
                    prev_word = b["prev_word"]
                    if USE_CUDA:
                        prev_word = prev_word.cuda()
                    parent_h = b["parent_h"]

                cur_s = list()
                predictions = list()
                for model_i in range(num_models):
                    curr_c, curr_h = decoders[model_i](prev_word, s[model_i][0], s[model_i][1],
                                                       parent_h[model_i], sibling_state[model_i])
                    cur_s.append((curr_c, curr_h))
                    attention_inputs = all_encoder_outputs[model_i][2]
                    prediction = attention_decoders[model_i](attention_inputs[0], curr_h, attention_inputs[1])
                    predictions.append(torch.exp(prediction))
                prediction = torch.mean(torch.stack(predictions, dim=0), dim=0)

                s = cur_s
                b["q"][head - 1]["s"] = s
                b["sibling_state"] = sibling_state
                b["parent_h"] = parent_h
                b["prev_word"] = prev_word

                topk_v, topk_i = torch.topk(prediction[0], beam_size)
                for value, index in zip(topk_v, topk_i):
                    new_b = beam_copy(b)
                    prev_word = [index.detach().cpu().numpy().item()]
                    new_b["prev_word"] = torch.LongTensor(prev_word).clone()
                    new_b["score"] += value.detach().cpu().numpy()
                    new_b["score_length"] += 1.0
                    s = new_b["q"][head - 1]["s"]

                    queue_decode = new_b["q"]
                    t = queue_decode[head - 1]["t"]

                    if int(prev_word[0]) == output_lang.word2index['<E>'] or t.num_children >= max_length:
                        new_b["head"] = head + 1
                        if new_b["head"] > len(new_b["q"]) or new_b["head"] > max_length:
                            new_b["head_done"] = True
                        else:
                            new_b["child"] = 1
                    elif int(prev_word[0]) == output_lang.word2index['<IE>']:
                        queue_decode.append(
                            {"s": [(ss[0].clone(), ss[1].clone()) for ss in s],
                             "parent": head, "child_index": i_child, "t": Tree()})
                        t.add_child(int(prev_word[0]))
                        new_b["child"] = i_child + 1
                    else:
                        t.add_child(int(prev_word[0]))
                        new_b["child"] = i_child + 1

                    new_beams.append(new_b)
            else:
                new_beams.append(b)

        beams = sorted(new_beams, key=lambda x: x["score"] / x["score_length"], reverse=True)[:beam_size]

    queue_decode = beams[0]["q"]
    for i in range(len(queue_decode) - 1, 0, -1):
        cur = queue_decode[i]
        queue_decode[cur["parent"] - 1]["t"].children[cur["child_index"] - 1] = cur["t"]

    return queue_decode[0]["t"].flatten(output_lang)


def topdown_train_tree(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch,
                       generate_nums, encoder, predict, generate, encoder_optimizer, predict_optimizer,
                       generate_optimizer, output_lang, num_pos, english=False):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)

    unk = output_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)

    target = torch.LongTensor(target_batch).transpose(0, 1)

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    encoder.train()
    predict.train()
    generate.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_var, input_length)
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_target_length = max(target_length)

    all_node_outputs = []
    # all_leafs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)

    num_start = output_lang.num_start
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        for idx, l, r, node_stack, i in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                            node_stacks, target[t].tolist()):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    loss = masked_cross_entropy(all_node_outputs, target, target_length)
    # loss = loss_0 + loss_1
    loss.backward()
    # clip the grad
    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


def topdown_evaluate_tree(input_batch, input_length, generate_nums, encoder, predict, generate, output_lang, num_pos,
                          beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)

    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_var, [input_length])

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.hidden_size)
    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            # leaf = p_leaf[:, 0].unsqueeze(1)
            # repeat_dims = [1] * leaf.dim()
            # repeat_dims[1] = op.size(1)
            # leaf = leaf.repeat(*repeat_dims)
            #
            # non_leaf = p_leaf[:, 1].unsqueeze(1)
            # repeat_dims = [1] * non_leaf.dim()
            # repeat_dims[1] = num_score.size(1)
            # non_leaf = non_leaf.repeat(*repeat_dims)
            #
            # p_leaf = torch.cat((leaf, non_leaf), dim=1)
            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            # out_score = p_leaf * out_score

            topv, topi = out_score.topk(beam_size)

            # is_leaf = int(topi[0])
            # if is_leaf:
            #     topv, topi = op.topk(1)
            #     out_token = int(topi[0])
            # else:
            #     topv, topi = num_score.topk(1)
            #     out_token = int(topi[0]) + num_start

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, embeddings_stacks, left_childs,
                                              current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0].out


def allocate_group_num(num_index_retreive, input_len, window_size=3):
    group_num = []
    for i in range(len(input_len)):
        # num_index_retreive[i] # [2, 11] --> [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, ~~~ last 3 worsd]
        group_num_tmp = []
        for num_idx in num_index_retreive[i] + [input_len[i] - 1]:
            group_num_tmp.extend([j for j in range(num_idx-window_size, num_idx+window_size+1) if j > 0 and j < input_len[i]])
        group_num_tmp = list(set(group_num_tmp))
        group_num.append(group_num_tmp)

    return group_num