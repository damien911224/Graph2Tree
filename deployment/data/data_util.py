#-*- coding:utf-8 -*-
import random
import json
import copy
import re
import numpy as np
import codecs

import torch.utils.data

from src.pre_data import Lang, OutputLang, indexes_from_sentence

import math
from data.parsing_dict import *


def decode(korean_num):
    decode_result = []
    result = 0
    temp_result = 0
    index = 0
    number_type = None

    for word in korean_num.split():
        if word in number_types:
            number_type = number_types.get(word)
        elif word.isdigit():
            result = int(word)

    if result > 0:
        if number_type is not None:
            return str(result) + number_type
        else:
            return result

    float_dividing = korean_num.split("점")
    float_result = ""
    if len(float_dividing) == 2:
        korean_num = float_dividing[0]
        float_num = float_dividing[1]
        for c in float_num:
            for float_num, float_value in float_nums:
                if c == float_num:
                    float_result += str(float_value)
                    break
        if len(float_result) == 0:
            float_result = 0.0
        else:
            float_result = float("0." + float_result)
    else:
        float_result = 0.0

    while index < len(korean_num):
        for number, true_value in numbers:
            if index + len(number) <= len(korean_num):
                if korean_num[index:index + len(number)] == number:
                    decode_result.append((true_value, math.log10(true_value).is_integer()))
                    if len(number) == 2:
                        index += 1
                    break
        index += 1

    for index, (number, is_natural) in enumerate(decode_result):
        if is_natural:
            if math.log10(number) > 3 and (math.log10(number) - 4) % 4 == 0:
                result += temp_result * number
                temp_result = 0

            elif index - 1 >= 0:
                if not decode_result[index - 1][1]:
                    temp_result += number * decode_result[index - 1][0]
                else:
                    temp_result += number
            else:
                temp_result += number

        else:
            if index + 1 == len(decode_result):
                temp_result += number
            elif not decode_result[index + 1][1]:
                temp_result += number
            elif math.log10(decode_result[index + 1][0]) > 3 and (math.log10(decode_result[index + 1][0]) - 4) % 4 == 0:
                temp_result += number

    result += temp_result

    if float_result != 0.0:
        result += float_result

    if number_type is not None:
        result = str(result) + number_type

    return result


def read_json_mac(path):
    # this is for mac
    with codecs.open(path, 'r', encoding="utf-8-sig") as f:
        file = json.load(f)
    return file


def read_json(path):
    with open(path, 'r', encoding='utf8') as f:
        file = json.load(f)
    return file


def load_data(path):
    questions = read_json_mac(path)
    return questions


def transfer_num_n_equation(data):
    pairs = {}
    generate_nums = {}
    copy_nums = 0

    for key in data:
        d = data[key]
        input_seq = d['question']
        nums = d['QL']
        names = d['NL']

        if copy_nums < len(nums):
            copy_nums = len(nums)

        num_pos = []
        for i, j in enumerate(input_seq):
            if "NUM" in j:
                num_pos.append(i)

            # output format of original code is
            # pairs[key] = (input_seq, output_seg, nums, num_pos)
        pairs[key] = (input_seq, None, nums, num_pos, names)

    return pairs, copy_nums


def prepare_infer_data(pairs, trim_min_count):
    input_lang = Lang()
    output_lang = OutputLang()
    test_pairs = []
    for idx in pairs:
        # number index
        pair = pairs[idx]
        if pair[-1]:
            input_lang.add_sen_to_vocab(pair[0])
    input_lang.build_input_lang(trim_min_count)

    for idx in pairs:
        pair = pairs[idx]
        input_cell = indexes_from_sentence(input_lang, pair[0])

        # inputcell, length_of_input, num_list, number_position, NL list
        test_pairs.append((input_cell, len(input_cell),
                          pair[2], pair[3], pair[4], str(idx)))

    return input_lang, output_lang, test_pairs


def load_data(filename):
    f = read_json_mac(filename)


def h2i(hangeul):
    hangeul = hangeul.strip()
    result = decode(hangeul)

    return result


def QL2Str(QL):
    result = '['
    for i in range(len(QL)):
        result += str(QL[i])
        if i != len(QL) - 1:
            result += ', '
    result += ']'
    return result

def diff_num_list(q_num, a, b):
    import re
    result = ''
    p = "\\d+\/\\d+"
    p = re.compile(p)
    if len(a) is not len(b):
        result = str(q_num) + " -> auto : " + str(a) + " user-define : " + str(list(map(str, b))) + '\n'
        return result
    for i in range(len(a)):
        A = p.match(str(a[i]))
        B = p.match(str(b[i]))
        if str(a[i]) != str(b[i]):
            if A is None and B is None:
                if float(a[i]) != float(b[i]):
                    result = str(q_num) + " -> auto : " + str(a) + " user-define : " + str(list(map(str, b))) + '\n'
            else:
                result = str(q_num) + " -> auto : " + str(a) + " user-define : " + str(list(map(str, b))) + '\n'
    return result

def extract(input_name):
    from konlp.kma.klt2000 import klt2000
    import json
    import re
    k = klt2000()

    # full_txt = f.read()
    # obj = json.loads(full_txt)
    list_obj = {}

    obj = read_json_mac(input_name)
    for q_num in obj:
        sent = obj[q_num]['question']
        sent = sent.replace(',', '')
        for kw in range(len(change_list1)):
            p = "\s" + change_list1[kw] + "\s"
            sent = re.sub(p, ' ' + target_list1[kw] + ' ', sent)
        sw = "|".join(change_list2)
        fw = "|".join(change_list3)
        cl = []
        p = "\s(" + sw + ")(" + fw + ")\s"
        if re.search(p, sent) is not None:
            for catch in re.finditer(p, sent):
                cl.append(catch[0])
        for i in cl:
            breakFlag = False
            for a in range(len(change_list2)):
                for b in range(len(change_list3)):
                    tmp = ' ' + change_list2[a] + change_list3[b] + ' '
                    if i == tmp:
                        sent = sent.replace(i, ' ' + target_list2[a] + target_list3[b] + ' ')
                        breakFlag = True
                        break
                if breakFlag:
                    break
        fw = "[" + "".join(change_list4) + "]"
        fw2 = "[" + "".join(change_list5) + "]"
        fw3 = "[" + "".join(change_list6) + "]"
        fw4 = "[" + "".join(unit) + "]"
        p = "(" + fw2 + "?" + fw3 + ")*" + fw + '?'
        # case1 : 삼백이십삼, 이십, 구십이 (양옆 띄어쓰기 상관X)
        cl = []
        if re.search(p, sent) is not None:
            for catch in re.finditer(p, sent):
                if len(catch[0]) > 1:
                    cl.append(catch[0])
        # case2 : 일, 이, 삼, 사, 오 (양옆 띄어쓰기 O)
        cl2 = []
        p = "\s(" + "|".join(change_list7) + ")\s"
        if re.search(p, sent) is not None:
            for catch in re.finditer(p, sent):
                if len(catch[0]) > 1:
                    cl2.append(catch[0])
        # case3 : 하나의, 하나만 1로 치환
        cl3 = []
        p = "(하나)[만가에의는당를\s]"
        if re.search(p, sent) is not None:
            for catch in re.finditer(p, sent):
                if len(catch[0]) > 1:
                    cl3.append(catch[0])
        # case4 : 첫째, 둘째 -> 1, 2로 치환
        cl4 = []
        fw8 = "(" + "|".join(change_list8) + ")"
        p = fw8 + "째"
        if re.search(p, sent) is not None:
            for catch in re.finditer(p, sent):
                if len(catch[0]) > 1:
                    cl4.append(catch[0])
        # case5 : 한으로 시작하는 경우
        fw9 = "^[한]\s"
        p = fw9
        sent = re.sub(fw9, "1 ", sent)

        p = '0+'
        p = re.compile(p)
        for i in cl:
            hh = p.match(i)
            if hh is None:
                sent = sent.replace(i, str(h2i(i)))
        for i in cl2:
            hh = p.match(i)
            if hh is None:
                sent = sent.replace(i, ' ' + str(h2i(i)) + ' ')
        for i in cl3:
            hh = p.match(i)
            if hh is None:
                sent = sent.replace(i, '1')
        for i in cl4:
            hh = p.match(i)
            if hh is None:
                for j in range(len(change_list8)):
                    if change_list8[j] in i:
                        sent = sent.replace(i, target_list2[j] + '째')

        nl = []
        p = "^(" + "|".join(NL_list) + "|" + "|".join(name_list) + ")(" + "|".join(josa_list) + ")"
        p2 = "\s(" + "|".join(NL_list) + "|" + "|".join(name_list) + ")(" + "|".join(josa_list) + ")"
        pn = "(" + "|".join(NL_list) + ")"
        if re.search(p, sent) is not None:
            for catch in re.finditer(p, sent):
                for catch2 in re.finditer(pn, catch[0]):
                    nl.append(catch2[0])
        if re.search(p2, sent) is not None:
            for catch in re.finditer(p2, sent):
                for catch2 in re.finditer(pn, catch[0]):
                    nl.append(catch2[0])

        tmp_obj = {}
        tmp_obj['NL'] = nl

        ql = []
        p = "\\d+(\\.\\d+)?(\\/\\d+)?"
        if re.search(p, sent) is not None:
            for catch in re.finditer(p, sent):
                ql.append(catch[0])

        tmp_obj['QL'] = ql
        tmp_obj['question'] = re.sub(p, "NUM", sent).split()
        try:
            tmp_obj['answer'] = obj[q_num]['lequation']
        except:
            pass
        list_obj[str(q_num)] = tmp_obj


    return list_obj
