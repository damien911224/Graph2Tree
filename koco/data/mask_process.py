from pyaichtools import Reducer
from tqdm import tqdm
import json

test_reducer = Reducer("data", debug=False)
#from pyaichtools import DefaultCfg, Converter
#temp_converter = Converter(DefaultCfg, debug=False)

with open("data/dummy.json") as problem:
	problem_json = json.load(problem)

mask_dict = {}
cnt =0  
for id, problem in tqdm(problem_json.items()):
	#label_seq = temp_converter.encode(problem['lequation'])
	label_seq = problem["lequation"]


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
		mask_dict[id] = mask_per_level
	except:
		cnt+=1
		print("{} pass".format(id)) 

	

print(cnt)
with open("data/mask_processed.json", "w") as dummy_file:
	json.dump(mask_dict, dummy_file)