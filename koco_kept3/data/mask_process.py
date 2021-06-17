from pyaichtools import Reducer, Converter, DefaultCfg
from tqdm import tqdm
import json

test_reducer = Reducer("data", debug=False)
test_converter = Converter(DefaultCfg, debug=True)
#from pyaichtools import DefaultCfg, Converter
#temp_converter = Converter(DefaultCfg, debug=False)

with open("data/dummy.json", "r", encoding="utf-8") as problem:
	problem_json = json.load(problem)

mask_dict = {}
cnt =0  
for id, problem in tqdm(problem_json.items()):
	#label_seq = temp_converter.encode(problem['lequation'])
	eq = problem["equation"]
	label_seq = test_converter.encode(eq)
	problem["lequation"] = label_seq
	problem_json[id] = problem

	queue = [[label_seq, None, None]]
	curr_id = 0
	mask_per_level = {}
	while len(queue) != curr_id:
		curr_node = queue[curr_id]
		curr_node.extend([curr_id])
		mask, new_nodes = test_reducer.test_label_mask(*curr_node)
		queue.extend(new_nodes)
		mask_per_level[curr_id] = mask
		curr_id += 1
	problem_json[id]["mask"] = mask_per_level
	"""
	try:
		queue = [[label_seq, None, None]]
		curr_id = 0
		mask_per_level = {}
		while len(queue) != curr_id:
			curr_node = queue[curr_id]
			curr_node.extend([curr_id])
			mask, new_nodes = test_reducer.test_label_mask(*curr_node)
			queue.extend(new_nodes)
			mask_per_level[curr_id] = mask
			curr_id += 1
		problem_json[id]["mask"] = mask_per_level
	except:
		cnt+=1
		problem_json[id]["mask"] = {}
		print("{} pass".format(id)) 
	"""
	

print(cnt)
with open("data/mask_flatten_dummy.json", "w") as dummy_file:
	json.dump(problem_json, dummy_file)