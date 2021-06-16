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
                '둘', '셋', '넷', \
                '한', \
                '스무', "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔" \
]
change_list2 = ["열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"]
change_list3 = ["한", '두', '세', '네', '다섯', '여섯', '일곱', '여덟', '아홉']
target_list1 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', \
            '2', '3', '4', \
            '1', \
            '20', '30', '40', '50', '60', '70', '80', '90'
]
target_list2 = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
target_list3 = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

change_list4 = ['일', '이', '삼', '사', '오', '육', '칠', '팔', '구']
change_list5 = ['이', '삼', '사', '오', '육', '칠', '팔', '구']
change_list6 = ['십', '백']
change_list7 = ['일', '이', '삼', '사', '오', '육', '칠', '팔', '구', '십', '백', '천', '만']
change_list8 = ['첫', '둘', '셋', '넷', '다섯', '여섯', '일곱', '여덟', '아홉']
unit=['kg','g','L','ml','mL','cm','mm','m','t','쪽','권','개입','개','명','원','묶음','단','모','세트','다스','병','장','박스','봉지','팩','줄','망','포','말','캔','판','자루','가마니','통']
target_list4 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
target_list5 = [10, 100, 1000, 10000]

NL_list = [ 
    '\(가\)', '\\(나\)', '\(다\)', '\(라\)', '\(마\)', '\(바\)', '\(사\)', '\(아\)', '\(자\)', '\(차\)', '\(카\)', '\(타\)', '\(파\)', '\(하\)', \
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'T', 'S', 'U', 'V', 'W', 'X', 'Y', 'Z', \
    '정국', '지민', '석진', '태형', '남준', '윤기', '호석', '민영', '유정', '은지', '유나', \
    '꼴찌', \
    '앞', '옆', '뒤', '가로', '세로', '왼쪽', '오른쪽', \
    '오전', '오후', \
    '월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일', \
    '아버지', '어머니', '할아버지', '할머니', '손자', '손녀', '조카', '이모', '삼촌', '동생', '누나', '오빠', \
    '손가락', '발가락', '팔', '다리', \
    '암컷', '수컷', '암탉', '수탉', '여학생', '남학생', '여차', '남자', \
    '흰색', '검은색', '파란색', '노란색', '초록색', '보라색', '빨간색', '주황색', '파란색', '남색', \
    '오리', '닭', '토끼', '물고기', '고래', '거위', '달팽이', '개구리', '강아지', '고양이', '비둘기', '병아리', '하마' \
    '연어', \
    '장미', '백합', '튤립', '카네이션', '국화', '화분', '화단', '꽃병', \
    '배구공', '농구공', '축구공', '탁구공', '줄넘기', '달리기', '수영', '시합', \
    '사과', '배', '감', '귤', '포도', '수박', '바나나', \
    '토마토', '무', '당근', '오이', '배추', \
    '나무', \
    '사탕', '김밥', '빵', '라면', '과자', '음료수', '주스', '우유', '달걀', '쌀', \
    '연필', '색연필', '지우개', '공책', '도화지', '색종이', '풀', '테이프', '바둑돌', '구슬', '상자', '나무토막', '장난감', '책장', '책꽂이', \
    '서점', '마트', '문구점', '집', '학교', '수영장', '교실', '도서관', '박물관', '운동장', '주차장', '정류장', '아파트', '농장', '강당', \
    '비행기', '자동차', '트럭', '배', '자전거', '오토바이', '기차', '버스', '엘리베이터', \
    '페인트', '벽', '천장', '문', '울타리'
]
name_list = ['정국이', '지민이', '석진이', '태형이', '남준이', '호석이', '민영이', '유정이']
josa_list = [
    '이', '가', '께서', '다', '을', '를', '의', '에' ,'로', '와', '과', '보다', '하고', '이랑', '랑', '이며', '며', '나', '만', '밖에', '뿐', '도', '조차', '마저', '까지', '이나', '나', \
    '이든지', '든지', '이나마', '나마', '이라도', '라도', '\s', '은', '는'
]
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

def extract(input_name, ans_name, output_name):
    from konlp.kma.klt2000 import klt2000 
    import json
    import re

    ans_obj = {}
    with open(ans_name, 'r', encoding='utf8') as g:
        ans_txt = g.read()
        ans_obj = json.loads(ans_txt)

    with open(input_name, 'r', encoding='utf8') as f:
        full_txt = f.read()
        obj = json.loads(full_txt)
        list_obj = {}
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
            list_obj[str(q_num)] = tmp_obj
        with open(output_name, 'w', encoding='utf8') as g:
            e = open('error_list.txt', 'w')
            diff_txt = ''
            main_obj = []
            error_list = []
            for q_num in list_obj:
                tmp = {}
                id = q_num
                try:
                    num_list = list_obj[str(q_num)]['QL']
                    ud_num_list = ans_obj[str(q_num)]['QL']
                    diff_txt += diff_num_list(str(q_num), num_list, ud_num_list)
                    noun_list = list_obj[str(q_num)]['NL']
                    new_text = obj[q_num]['question']
                    new_equation = ans_obj[q_num]['equation']
                    solution = ans_obj[q_num]['answer']
                    
                    num_list_str = re.sub("\'", "", str(num_list))
                    QL = "QL = " + num_list_str
                    NL = "NL = " + str(noun_list)

                    tmp['id'] = id
                    tmp['QL_code'] = QL
                    tmp['NL_code'] = NL
                    tmp['new_text'] = new_text
                    tmp['new_equation'] = new_equation
                    tmp['lSolution'] = solution
                    main_obj.append(tmp)
                except Exception as ex:
                    error_tmp = {}
                    error_tmp['q_num'] = q_num
                    error_tmp['error'] = str(ex)
                    error_list.append(error_tmp)
                    continue

            g.write(json.dumps(main_obj, indent=4, ensure_ascii=False))
            e.write(json.dumps(error_list, indent=4))
            with open('diff.txt', 'w', encoding='utf8') as d:
                d.write(diff_txt)
            e.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Input File Name')
    parser.add_argument('--prob', required=True, default='problem.json', help='Problem File Name')
    parser.add_argument('--ans', required=True, default='', help='Answer File Name')
    parser.add_argument('--data', required=False, default='data/output.json', help='Output File Name')
    output_name = "output.json"
    args = parser.parse_args()
    extract(args.prob, args.ans, args.data)