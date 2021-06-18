from collections import defaultdict
from os import write
from yacs.config import CfgNode as CN
from pyaichtools import Converter
from pyaichtools import DefaultCfg as cfg
import json
import libcst as cst
import tqdm
import sys
import math, itertools

def load_json(file_path):
    with open(file_path, 'r') as json_file:
        json_data = json.load(json_file)
    return json_data


header_path = 'test/src/header.py'
footer_path = 'test/src/footer.py'
body_path = 'test/src/body.py'
ql_path = 'test/src/ql.py'
answer_path = 'answer_combined.json'
problem_path = 'problem_combined_refined.json'
write_path = 'koco_03.json'

cfg.header_path = header_path
cfg.footer_path = footer_path
cfg.ql_path = ql_path
cfg.SPT = '/'

debug_converter = Converter(cfg, debug=True)
test_converter = Converter(cfg, debug=False)
problem_json = load_json("data/koco_04.json")


cnt = 0
#QL = range(1, 4000, 3)
CONST = "const{} = {}"
const_list = [CONST.format(i, i) for i in range(0, 2000)]
exec(str.join('\n',const_list))

target_list = ['1986', '1997', '2000', '2004', '2005', '2051', '2264', '2351', '2353', '2354', '2357', '2361', '2362', '2363', '2364', '2367', '2368', '2370', '2378', '2380', '2381', '2382', '2383', '2385', '2386', '2387', '2388', '2390', '2402', '2403', '2404', '2406', '2421', '2497', '2505', '2507', '2512', '2514']

for k, v in tqdm.tqdm(problem_json.items()):

	if k in target_list:
		continue

	try:
		test_converter.encode(v["equation"])
		v["lequation"] = debug_converter.encode(v["equation"])
		QL = v["QL"]
		generated = debug_converter.decode(v["lequation"])
		exec(v["equation"])
		origin_result = result
		exec(generated)
		assert origin_result == result
	except:
		print(sys.exc_info(), k)
		v["lequation"] = ""
		cnt+=1



print(cnt)
with open("data/koco_04_processed.json", "w") as prob:
	json.dump(problem_json, prob)