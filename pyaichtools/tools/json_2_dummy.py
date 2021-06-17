from os import write
from yacs.config import CfgNode as CN
from pyaichtools import Converter
from pyaichtools import DefaultCfg as cfg
import json
import libcst as cst
import tqdm

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
problem_json = load_json("data/koco_03.json")


cnt = 0
for k, v in tqdm.tqdm(problem_json.items()):
	try:
		test_converter.encode(v["equation"])
		v["lequation"] = debug_converter.encode(v["equation"])
	except:
		v["lequation"] = ""
		cnt+=1


print(cnt)
with open("data/koco_03.json", "w") as prob:
	json.dump(problem_json, prob)