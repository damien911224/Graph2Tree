from collections import defaultdict
import os
import json
import typing
import libcst as cst
import inspect
import sys
from .utils import *
from treelib import Tree, Node				
import copy
from operator import itemgetter
import time

class Reducer():
	def __init__(self, label_root_path, cst_class_tree_path=None, debug=False):
		with open(os.path.join(label_root_path, "label_dict.json")) as ld_file:
			self.label_dict = json.load(ld_file)
		
		with open(os.path.join(label_root_path, "reverse_label_dict.json")) as rld_file:
			self.reverse_label_dict = json.load(rld_file)
		
		self.label_dict["None"] = "None"
		self.reverse_label_dict["None"] = "None"

		self.range_per_pi = {
			"math": list(range(self.reverse_label_dict["math.acos"], self.reverse_label_dict["math.trunc"]+1)),
			"itertools": list(range(self.reverse_label_dict["itertools.__loader__"], self.reverse_label_dict["itertools.zip_longest"]+1)),
			"QL": list(range(self.reverse_label_dict["QL[0]"], self.reverse_label_dict["QL[19]"]+1)),
			"NL": list(range(self.reverse_label_dict["NL[0]"], self.reverse_label_dict["NL[49]"]+1)),
			"var": list(range(self.reverse_label_dict["var0"], self.reverse_label_dict["const1000"]+1)),
			"signal": list(range(self.reverse_label_dict["<S>"], self.reverse_label_dict["<IE>"]+1))
		}
		self.reverse_range_per_pi = defaultdict(lambda:"None")
		for k, v in self.range_per_pi.items():
			for v_ele in v:
				self.reverse_range_per_pi[v_ele] = k

		self.get_child_node = lambda x: [attr for attr in getattr(cst, x).__dict__['__annotations__'].items() if attr[0] in LIBCST_INTERST_ATTR]
		self.cst_need_child = lambda x: '__annotations__' in getattr(cst, x).__dict__

		if debug:
			self.id_to_block_label = lambda x: self.label_dict[str(x)]
		else:
			self.id_to_block_label = lambda x: x

		self.name_to_pi = {
			self.reverse_label_dict["Name"]: ["var"],
			self.reverse_label_dict["Subscript"]: ["QL", "NL"],
			self.reverse_label_dict["Attribute"]: ["math", "itertools"]
		}

		if cst_class_tree_path is None:
			self.cst_class_tree = self.build_cst_class_tree()

			with open(os.path.join(label_root_path, 'cst_tree_dict.json'), "w") as tree_file:
				json.dump(self.cst_class_tree.to_dict(), tree_file)
		else:
			with open(cst_class_tree_path) as tree_file:
				self.cst_class_tree = json.load(tree_file)
			#for k, v in self.cst_class_tree.items():
			#	self.cst_class_tree[k] = 

		self.hard_except_label = ["BaseString", "BaseNumber", "BaseDict", "BaseSet", "ClassDef", "FunctionDef", "With", "BaseFormattedStringContent", "Try"]
		self.hard_except_label = [self.cst_class_tree.subtree(x).leaves() for x in self.hard_except_label]
		temp_label = []
		for x in self.hard_except_label:
			for node in x:
				temp_label.append(self.reverse_label_dict[node.tag])
	
		self.hard_except_label = temp_label
		if hasattr(typing, '_GenericAlias'):
			self.sequence_test = lambda x: type(x).__name__ == '_GenericAlias' and x._name == 'Sequence'
			self.union_test = lambda x :type(x).__name__ == '_GenericAlias' and x._name == None and len(x.__args__) > 1
		else:
			self.sequence_test = lambda x : (type(x) == typing.Sequence) or ( 'Sequence' == x.__name__ if hasattr(x, '__name__') else False)
			self.union_test = lambda x: (type(x) == typing.Union) or ('_Union' == type(x).__name__ if hasattr(type(x), '__name__') else False)


	def build_cst_class_tree(self):
		cst_classes = {cst_class: getattr(cst, cst_class) for cst_class in dir(sys.modules["libcst"]) if inspect.isclass(getattr(cst, cst_class))}
		cst_tree_dict = {}
		for _cst_class in list(cst_classes.values()):
			curr_tree = Tree()
			curr_tree.create_node(tag=_cst_class.__name__, identifier=_cst_class.__name__)
			cst_tree_dict[_cst_class.__name__] = curr_tree
		
		cst_count = 0

		while cst_count != len(cst_tree_dict):
			for curr_tree in list(cst_tree_dict.keys()):
				cst_count+=1
				if hasattr(cst_classes[curr_tree], '__base__'):
					curr_base = cst_classes[curr_tree].__base__.__name__ 
					if curr_base[0] == '_' and len(cst_classes[curr_tree].__bases__) > 1 and "libcst" in cst_classes[curr_tree].__bases__[1].__module__:
						curr_base = cst_classes[curr_tree].__bases__[1].__name__
					parent_tree_key = [_tree for _tree in list(cst_tree_dict.keys()) if curr_base in list(cst_tree_dict[_tree].expand_tree())]

					if len(parent_tree_key) != 0:
						parent_tree_key = parent_tree_key[0]
					elif 'libcst' in cst_classes[curr_tree].__base__.__module__.split('.'):
						new_tree = Tree()
						cst_classes[curr_base] = cst_classes[curr_tree].__base__
						new_tree.create_node(curr_base, curr_base)
						cst_tree_dict[curr_base] = new_tree
						parent_tree_key = curr_base
					else:
						continue

					curr_parent_tree = cst_tree_dict[parent_tree_key]
					curr_parent_tree.paste(curr_base, cst_tree_dict[curr_tree])
					#curr_parent_tree.add_node(cst_tree_dict[curr_tree].get_node(curr_tree), parent=curr_base)
					del cst_tree_dict[curr_tree]
					cst_count = 0
					break
		
		return cst_tree_dict["CSTNode"]
	
	def flatten_cst_type(self, direct_attr):
		direct_label = []
		assert type(direct_attr) is not str
		if self.sequence_test(direct_attr) or self.union_test(direct_attr):
			for attr_ele in direct_attr.__args__:
				if 'libcst' in attr_ele.__module__:
					direct_label.extend([self.reverse_label_dict[node.tag] for node in self.cst_class_tree.leaves(attr_ele.__name__)])
				if self.sequence_test(direct_attr) or self.union_test(direct_attr):
					direct_label.extend(self.flatten_cst_type(attr_ele))
		else:
			direct_label.extend([self.reverse_label_dict[node.tag] for node in self.cst_class_tree.leaves(direct_attr.__name__)])
				
		return list(set(direct_label))

	def get_candidate(self, parent_label, direct_attr, prev_pred_label):
		seq_label = []
		direct_label = []
		curr_latest_label = prev_pred_label[-1]
		curr_latest_label_child_num = 0

		for prl in prev_pred_label[::-1]:
			if hasattr(cst, prl):
				curr_latest_label = prl
				break
			elif self.reverse_range_per_pi[prl] not in ["signal", "None"]:
				curr_latest_label = prl
				break
			curr_latest_label_child_num += 1

		#6. Curren label is leaf node label predict End token only
		seq_label = [self.reverse_label_dict["<E>"]]

		direct_label = self.flatten_cst_type(direct_attr)

		if hasattr(cst, curr_latest_label):
			# 2. If curr_latest_label is CST Node, check if it has sufficient child
			if len(self.get_child_node(curr_latest_label)) > curr_latest_label_child_num:
				#3. has insufficient child
				seq_label = [self.reverse_label_dict["<IE>"]]
			elif self.sequence_test(direct_attr):
				#4. has sufficient child and parent label require list of child
				seq_label += direct_label
		elif curr_latest_label == "<S>" or curr_latest_label == "<IS>":
			#5. Current layer has just begin, need to predict label by direct attr
			seq_label = direct_label
		
		seq_label = self.remove_hmn(seq_label)

		return seq_label
	
	def remove_hmn(self, candidate_list):
		temp_candidate_list =  []
		for hmn in list(self.name_to_pi.keys()):
			if hmn in candidate_list:
				candidate_list.remove(hmn)
				for pi in self.name_to_pi[hmn]:
					curr_pi_label = self.range_per_pi[pi]
					temp_candidate_list.extend(curr_pi_label)

		for hard_except_label in self.hard_except_label:
			if hard_except_label in candidate_list:
				candidate_list.remove(hard_except_label)

		return candidate_list + temp_candidate_list 
		
	def reduce_out(self, parent_id_list, parent_child_idx, prev_pred_list):

		str_lister = lambda x: [x] if type(x) == str else list(x)
		parent_label = str_lister(itemgetter(*[str(v) for v in parent_id_list])(self.label_dict))
		tuple_lister =  lambda x: [x] if type(x) != tuple else list(x)
		prev_pred_label = [tuple_lister(itemgetter(*[str(v) for v in inner_v])(self.label_dict)) for inner_v in prev_pred_list]
		#pi_label = list(itemgetter(*curr_id_list)(self.reverse_range_per_pi))

		# predicted cst node which need child node, return possible prediction as dictionary

		direct_attr_list = []
		for pci, p_label in zip(parent_child_idx, parent_label):
			cnt = 0
			curr_attr = None
			if p_label is "None":
				direct_attr_list.append(cst.Module)
				continue
			assert pci is not None
			curr_child_node = self.get_child_node(p_label)
			for v in sorted(curr_child_node, key=lambda x:x[0]):
				if cnt == pci:
					curr_attr = v[1]
					break
				cnt += 1
			direct_attr_list.append(curr_attr)

		start = time.perf_counter()

		candidate = [
		    self.get_candidate(pl, al, ppl)
		    for pl, al, ppl in zip(parent_label, direct_attr_list, prev_pred_label)
		]

		return candidate

	def test_label_mask(self, label_seq, parent_label=None, child_idx=None, queue_id=0):
		label_seq.insert(0, self.reverse_label_dict["<S>" if queue_id == 0 else "<IS>"])
		label_seq.append(self.reverse_label_dict["<E>"])
		parent_label_list = []
		prev_pred_list = []
		child_idx_list = []

		for id, node in enumerate(label_seq):
			if id == 0:
				continue
			parent_label_list.append(parent_label if parent_label is not None else "None")
			child_idx_list.append(child_idx)
			prev_pred_list.append([x if type(x) is not list else self.reverse_label_dict["<IE>"] for x in label_seq[:id]])
		
		
		mask = self.reduce_out(parent_label_list, child_idx_list, prev_pred_list)
		
		for id, node in enumerate(label_seq):
			if id == 0:
				continue
			node_label = node if type(node) is not list else self.reverse_label_dict["<IE>"]
			temp_mask = [self.id_to_block_label(x) for x in mask[id-1]]
			print(self.id_to_block_label(node_label), temp_mask)
			if node_label not in mask[id-1]:
				raise Exception("Non predictable node from mask")
		
		next_list = []

		latest_label_diff = 0
		latest_label = None
		for id, node in enumerate(label_seq):
			if type(node) == list:
				next_list.append([node, latest_label, latest_label_diff-1])
			elif hasattr(cst, self.label_dict[str(node)]):
				latest_label_diff = 0
				latest_label = node
			latest_label_diff += 1

		return mask, next_list


if __name__ == '__main__':
	test_reducer = Reducer("label", debug=False)
	from pyaichtools import DefaultCfg, Converter
	temp_converter = Converter(DefaultCfg, debug=False)
	body_path = 'test/src/body.py'
	label_seq = temp_converter.encode(body_path)

	"""
	parent_label = [test_reducer.reverse_label_dict[v] for v in ["AugAssign","If"]]
	prev_label = [[test_reducer.reverse_label_dict[v] for v in inner_v] for inner_v in [["<IE>","BinaryOperation","<IE>","<IE>","<IE>"], ["<IE>"]] ]
	child_idx = [1, 2]
	"""

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

	with open("dummy.json", "w") as dummy_file:
		json.dump(mask_per_level, dummy_file)