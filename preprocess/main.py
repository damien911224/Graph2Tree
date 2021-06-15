#-*- coding:utf-8 -*-

import math

"""
Developed by Junseong Kim, Atlas Guide
codertimo@goodatlas.com / github.com/codertimo
Korean to number
"""

numbers = [
    ("스물", 20),
    ("서른", 30),
    ("마흔", 40),
    ("쉰", 50),
    ("예순", 60),
    ("일흔", 70),
    ("여든", 80),
    ("아흔", 90),
    ("하나", 1),
    ("한", 1),
    ("두", 2),
    ("둘", 2),
    ("세", 3),
    ("셋", 3),
    ("네", 4),
    ("넷", 4),
    ("다섯", 5),
    ("여섯", 6),
    ("일곱", 7),
    ("여덟", 8),
    ("여덜", 8),
    ("아홉", 9),
    ("일", 1),
    ("이", 2),
    ("삼", 3),
    ("사", 4),
    ("오", 5),
    ("육", 6),
    ("칠", 7),
    ("팔", 8),
    ("구", 9),
    ("열", 10),
    ("십", 10),
    ("백", 100),
    ("천", 1000),
    ("만", 10000),
    ("억", 100000000),
    ("조", 1000000000000),
    ("경", 10000000000000000),
    ("해", 100000000000000000000),
]

number_types = {
    "키로": "kg",
    "키로그램": "kg",
    "킬로": "kg",
    "킬로그램": "kg",
    '킬로그람': "kg",
    "그램": "g",
    "그람": "g",
    "리터": "L",
    "밀리리터": "mL",
    "미리리터": "mL",
    "미리": "mL",
    "밀리": "mL",
    "센치미터": "cm",
    "센티미터": "cm",
    "밀리미터": "mm",
    "미터": "m",
    "개입": "개입",
    "개": "개",
    "명": "명",
    "원": "원",
    "묶음": "묶음",
    "단": "단",
    "모": "모",
    "세트": "세트",
    "병": "병",
    "장": "장",
    "박스": "박스",
    "봉지": "봉지",
    "팩": "팩",
    "줄": "줄",
    "망": "망",
    "포": "포",
    "말": "말",
    "캔": "캔",
    "판": "판",
    "자루": "자루",
    "가마니": "가마니",
    "통": "통",
    "다스": "다스",
    "권":"권",
    "쪽":"쪽"
}

float_nums = [
    ("일", 1),
    ("이", 2),
    ("삼", 3),
    ("사", 4),
    ("오", 5),
    ("육", 6),
    ("칠", 7),
    ("팔", 8),
    ("구", 9)
]


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

change_list1 = ['첫', '두', '세', '네', '다섯', '여섯', '일곱', '여덟', '아홉', '열', \
                '하나', '둘', '셋', '넷', \
                '한', \
                '스무', "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔" \
]
change_list2 = ["열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"]
change_list3 = ["한", '두', '세', '네', '다섯', '여섯', '일곱', '여덟', '아홉']
target_list1 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', \
            '1', '2', '3', '4', \
            '1', \
            '20', '30', '40', '50', '60', '70', '80', '90'
]
target_list2 = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
target_list3 = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

change_list4 = ['일', '이', '삼', '사', '오', '육', '칠', '팔', '구']
change_list5 = ['이', '삼', '사', '오', '육', '칠', '팔', '구']
change_list6 = ['십', '백', '천', '만']
change_list7 = ['일', '이', '삼', '사', '오', '육', '칠', '팔', '구', '십', '백', '천', '만']
unit=['kg','g','L','ml','mL','cm','mm','m','t','쪽','권','개입','개','명','원','묶음','단','모','세트','다스','병','장','박스','봉지','팩','줄','망','포','말','캔','판','자루','가마니','통']
target_list4 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
target_list5 = [10, 100, 1000, 10000]

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

def extract(input_name):
    from konlp.kma.klt2000 import klt2000 
    import json
    import re
    k = klt2000()

    with open(input_name, 'r', encoding='utf8') as f:
        full_txt = f.read()
        obj = json.loads(full_txt)
        list_obj = {}
        for q_num in obj:
            sent = obj[q_num]['question']
            sent = sent.replace(',', '')
            #sent = change_hangeul(sent)
            '''
            ql_candi = k.pos(sent)
            for ql_c in ql_candi:
                for idx in range(len(change_list)):
                    if change_list[idx] == ql_c[:-2] and (ql_c[-1] == 'K' or ql_c[-1] == 'N' or ql_c[-1] == 'W'):
                        sent = sent.replace(ql_c[:-2], target_list[idx])
            ql_candi = k.pos(sent)
            '''
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
            fw = "(" + "|".join(change_list4) + ")"
            fw2 = "(" + "|".join(change_list5) + ")"
            fw3 = "(" + "|".join(change_list6) + ")"
            fw4 = "(" + "|".join(unit) + ")"
            p = "(" + fw2 + "?" + fw3 + ")*" + fw + '?(\s |' + fw4 + ')'  
            cl = []
            if re.search(p, sent) is not None:
                for catch in re.finditer(p, sent):
                    if len(catch[0]) > 1:
                        cl.append(catch[0])
            cl2 = []
            p = "\s(" + "|".join(change_list7) + ")\s"
            if re.search(p, sent) is not None:
                for catch in re.finditer(p, sent):
                    if len(catch[0]) > 1:
                        cl2.append(catch[0])
            for i in cl:
                sent = sent.replace(i, str(h2i(i)))
            for i in cl2:
                sent = sent.replace(i, ' ' + str(h2i(i)) + ' ')

            nl = k.nouns(sent)
            tmp_obj = {}
            tmp_obj['NL'] = nl

            ql = []
            p = "\\d+(\\.\\d+)?"
            if re.search(p, sent) is not None:
                for catch in re.finditer(p, sent):
                    ql.append(catch[0])
                    
            tmp_obj['QL'] = ql
            list_obj[str(q_num)] = tmp_obj
        with open('qlnl_list.txt', 'w', encoding='utf8') as g:
            for q_num in list_obj:
                QL = list_obj[str(q_num)]['QL']
                NL = list_obj[str(q_num)]['NL']
                g.write(str(q_num) + ":\n")
                g.write("QL = ")
                g.write(QL2Str(QL) + '\n')
                g.write("NL = ")
                g.write(str(NL) + '\n\n')

                QL_index = []
                NL_index = []
                for i in range(len(QL)):
                    QL_index.append(str(i) + ': ' + str(QL[i]))
                for j in range(len(NL)):
                    NL_index.append(str(j) + ': ' + str(NL[j]))

                g.write("QL_index = ")
                g.write(QL2Str(QL_index) + '\n')
                g.write("NL_index = ")
                g.write(str(NL_index) + '\n\n')

def extract2(input_name, answer_name):
    from konlp.kma.klt2000 import klt2000 
    import json
    import re
    k = klt2000()
    h = open(answer_name, 'r', encoding='utf8') 
    full_ans = h.read()
    ans_obj = json.loads(full_ans)

    with open(input_name, 'r', encoding='utf8') as f:
        full_txt = f.read()
        obj = json.loads(full_txt)
        list_obj = {}
        for q_num in obj:
            sent = obj[q_num]['question']
            sent = sent.replace(',', '')
            #sent = change_hangeul(sent)
            ql_candi = k.pos(sent)
            for ql_c in ql_candi:
                for idx in range(len(change_list)):
                    if change_list[idx] == ql_c[:-2] and (ql_c[-1] == 'K' or ql_c[-1] == 'N'):
                        sent = sent.replace(ql_c[:-2], target_list[idx])
            ql_candi = k.pos(sent)
            nl = k.nouns(sent)
            tmp_obj = {}
            tmp_obj['NL'] = nl

            ql = []
            p = "\\d+(\\.\\d+)?"
            if re.search(p, sent) is not None:
                for catch in re.finditer(p, sent):
                    ql.append(catch[0])
                    
            '''
            for i in ql_candi:
                if '#' in i:
                    num = i[:-2]
                    num = str(float(num))
                    if num[-1] == '0' and num[-2] == '.':
                        num = num[:-2]
                    ql.append(num)
            '''
            tmp_obj['QL'] = ql
            list_obj[str(q_num)] = tmp_obj
            if q_num in ans_obj:
                if str(ans_obj[q_num]['answer'])[-1] == '0' and str(ans_obj[q_num]['answer'])[-2] == '.':
                    ans_obj[q_num]['answer'] = int(str(ans_obj[q_num]['answer'])[:-2])
                eq = ans_obj[q_num]['equation']
                eq_list = eq.split('=')
                eq_list[0] = 'result'
                if len(eq) < 2:
                    eq_list.append('')

                p = "\\d+(\\.\\d+)?"
                if re.search(p, eq_list[1]) is not None:
                    for catch in re.finditer(p, eq_list[1]):
                        if len(catch[0]) > 2 and catch[0][-1] == '0' and catch[0][-2] == '.':
                            eq_list[1] = eq_list[1].replace(catch[0], catch[0][:-2])
                for i in range(len(ql)):
                    if str(ql[i]) in eq_list[1]:
                        eq_list[1] = eq_list[1].replace(str(ql[i]), 'QL[' + str(i) + ']')
                eq = '='.join(eq_list)
                ans_obj[q_num]['equation'] = eq
        with open('ans.json', 'w', encoding='utf8') as a:
            a.write(json.dumps(ans_obj, indent=4))

        with open('qlnl_list.txt', 'w', encoding='utf8') as g:
            for q_num in list_obj:
                QL = list_obj[str(q_num)]['QL']
                NL = list_obj[str(q_num)]['NL']
                g.write(str(q_num) + ":\n")
                g.write("QL = ")
                g.write(QL2Str(QL) + '\n')
                g.write("NL = ")
                g.write(str(NL) + '\n\n')

                QL_index = []
                NL_index = []
                for i in range(len(QL)):
                    QL_index.append(str(i) + ': ' + str(QL[i]))
                for j in range(len(NL)):
                    NL_index.append(str(j) + ': ' + NL[j])

                g.write("QL_index = ")
                g.write(QL2Str(QL_index) + '\n')
                g.write("NL_index = ")
                g.write(str(NL_index) + '\n\n')
        h.close()
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Input File Name')
    parser.add_argument('--input', required=False, default='problem.json', help='Input File Name')
    parser.add_argument('--ans', required=False, default='', help='Answer File Name')
    args = parser.parse_args()
    if args.ans != '':
        extract(args.input)
    else:
        extract(args.input)