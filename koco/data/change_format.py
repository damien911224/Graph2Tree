import numpy as np
import os

import sys
import json
import csv
import re

def read_json(fn):
    with open(fn, 'r', encoding='utf-8-sig') as f:
        full_txt = f.read()
        json_data = json.loads(full_txt)

    return json_data

if __name__ == '__main__':

    fn_from = sys.argv[1]
    fn_from_label = sys.argv[2]
    fn_to = sys.argv[3]
    
    json_data = read_json(fn_from)
    json_data_label = read_json(fn_from_label)

    # print(json_data_label.keys())
    # print(json_data_label['496'])
    t = read_json("data/mawps_combine.json")
    print(t[0])

    converted_q = []

    p = "\\d{1,3}(\\.\\d{1,3})?"

    for k in json_data.keys():
        new_dict = {}
        jd = json_data[k]
        
        sent = jd['question']

        p = "\\d{1,3}(\\.\\d{1,3})?"

        numbers = []
        if re.search(p, sent) is not None:
            for catch in re.finditer(p, sent):
                numbers.append(catch[0])
        
        ques = sent
        for idx, num in enumerate(numbers):
            ques = ques.replace(num, "number{} ".format(idx), 1)

        eq, ans = json_data_label[k]['equation'], json_data_label[k]['answer']

        new_dict['new_text'] = sent
        new_dict['sQuestion'] = sent
        new_dict['num_list'] = numbers
        new_dict['lEquations'] = [eq]
        new_dict['lSolutions'] = [ans]

        converted_q.append(new_dict)

    with open(fn_to, 'w', encoding='utf-8') as make_file:
        json.dump(converted_q, make_file)

