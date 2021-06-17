from collections import defaultdict
import os
import libcst as cst
import sys
import inspect
import json
from treelib import Tree, plugins
from pyaichtools.utils import *
import typing
import re


class Converter:
    def __init__(self, cfg, debug=False):
        try:
            with open(cfg.header_path) as header_file:
                header_file = header_file.read()
                self.header = cst.parse_module(header_file)
        except:
            print("No header")

        try:
            with open(cfg.ql_path, encoding="utf-8") as ql_file:
                ql_file = ql_file.read()
                self.quality_list = cst.parse_module(ql_file)
        except:
            print("No ql")

        try:
            with open(cfg.gen_head_path) as gen_head_file:
                gen_head_file = gen_head_file.read()
                self.gen_head = cst.parse_module(gen_head_file)
        except:
            print("No ql")

        try:
            with open(cfg.footer_path) as footer_file:
                footer_file = footer_file.read()
                self.footer = cst.parse_module(footer_file)
        except:
            print("No ql")

        self.interest_attr_list = LIBCST_INTERST_ATTR
        self.var_list = [LABEL_PREFIX_INFO["VAR_PREFIX"].format(i) for i in range(cfg.var_range)] + ["result"]
        self.const_list = [LABEL_PREFIX_INFO["CONST_PREFIX"].format(i) for i in list(range(cfg.const_range)) + [100, 1000]]
        #self.tree_spt_list = ['nodest', 'nodeen', 'argst', 'argen']

        self.label_dict, self.reverse_label_dict, self.LABEL_LIMIT = \
         self.generate_label_dict(self.var_list, self.const_list) #self.tree_spt_list,)

        #if you want to generate new label dictionary, uncomment these lines
        with open('data/label_dict.json', 'w') as ld:
            json.dump(self.label_dict, ld)

        with open('data/reverse_label_dict.json', 'w') as rld:
            json.dump(self.reverse_label_dict, rld)

        self.SPT = cfg.SPT
        self.attach_code = lambda x: self.header.body + self.quality_list.body + x + self.footer.body
        self.debug = debug
        self.hard_code_label = ["Attribute", "Subscript", "Name"]

        self.cst_need_child = lambda x: '__annotations__' in getattr(cst, x).__dict__

        self.has_child_node = defaultdict(bool)

        self.get_child_node = lambda x: [attr for attr in getattr(cst, x).__dict__['__annotations__'].items() if attr[0] in LIBCST_INTERST_ATTR]

        for k in list(self.reverse_label_dict.keys()):
            if hasattr(cst, k) and self.cst_need_child(k):
                self.has_child_node[k] = len(self.get_child_node(k)) > 0

    def attach_gen_file(self, generated):
        self.gen_head.body[0].body.body.extend(generated.body)
        #self.gen_head.body[0].body.body = [base_frame[0]] + generated.body + [base_frame[1]]
        return self.gen_head

    def generate_label_dict(self, var_list, const_list):
        libcst_class_list = [
         '{}'.format(name)
         for name, obj in inspect.getmembers(sys.modules['libcst'])
         if inspect.isclass(obj)
        ]
        libcst_class_list.sort()

        math_func_list = [
         'math.{}'.format(name)
         for name, obj in inspect.getmembers(sys.modules['math'])
         if inspect.isbuiltin(obj)
        ]
        math_func_list.sort()

        itertools_class_list = [
         'itertools.{}'.format(name)
         for name, obj in inspect.getmembers(sys.modules['itertools'])
         if inspect.isclass(obj)
        ]
        itertools_class_list.sort()

        whole_label_list = libcst_class_list + math_func_list + itertools_class_list + var_list + const_list
        whole_label_list.extend(LABEL_PREFIX_INFO["ST_EN_PREFIX"])
        LABEL_LIMIT = len(whole_label_list)
        whole_label_list.extend([LABEL_PREFIX_INFO["QL_PREFIX"].format(i) for i in range(LABEL_PREFIX_INFO["MAX_QUANTITY_LEN"])])
        whole_label_list.extend([LABEL_PREFIX_INFO["NL_PREFIX"].format(i) for i in range(LABEL_PREFIX_INFO["MAX_NOUN_LEN"])])

        label_dict = {k: v for k, v in zip(range(len(whole_label_list)), whole_label_list)}
        reverse_label_dict = {k: v for k, v in zip(whole_label_list, range(len(whole_label_list)))}

        return label_dict, reverse_label_dict, LABEL_LIMIT

    def cst_to_tree(self, parsed_cst, cst_tree, parent_id=None, attr=None):

        if not hasattr(parsed_cst, '__module__'):
            curr_node = cst_tree.create_node(str.join(self.SPT, [attr, parsed_cst]), parent=parent_id)
            return
        elif attr is None:
            curr_node = cst_tree.create_node(type(parsed_cst).__name__, parent=parent_id)
        else:
            curr_node = cst_tree.create_node(str.join(self.SPT, [attr, type(parsed_cst).__name__]), parent=parent_id)

        curr_attr_list = dir(parsed_cst)
        for interest_attr in curr_attr_list:
            if interest_attr in self.interest_attr_list:
                if type(getattr(parsed_cst, interest_attr)) in [list ,tuple]:
                    for attr_ele in getattr(parsed_cst, interest_attr):
                        self.cst_to_tree(attr_ele, cst_tree, parent_id=curr_node.identifier, attr=interest_attr)
                else:
                    self.cst_to_tree(getattr(parsed_cst, interest_attr), cst_tree, parent_id=curr_node.identifier, attr=interest_attr)
        return cst_tree

    def tree_to_seq(self,ann_tree, seq=[]):
        curr_child = ann_tree.children(ann_tree.root)
        curr_tag = ann_tree.get_node(ann_tree.root).tag.split(self.SPT)
        if len(curr_child) == 0:
            curr_seq = ["nodest","nodeen",curr_tag[1],]
        else:
            curr_seq = ["nodest"]
            prev_attr = curr_child[0].tag.split(self.SPT)[0]
            curr_seq.append("argst")
            for child_node in curr_child:
                curr_attr = child_node.tag.split(self.SPT)[0]
                if prev_attr != curr_attr:
                    curr_seq.extend(["argen", "argst"])
                curr_seq = self.tree_to_seq(ann_tree.subtree(child_node.identifier), curr_seq)
            curr_seq.append("argen")
            curr_seq.append("nodeen")
            curr_seq.append(curr_tag[1])
        seq.extend(curr_seq)
        return seq

    def tree_to_list(self,ann_tree, seq=[],label_to_id=False):
        curr_child = ann_tree.children(ann_tree.root)
        curr_tag = ann_tree.get_node(ann_tree.root).tag.split(self.SPT)
        curr_seq = [self.label_ele(curr_tag[1], ann_tree, label_to_id)]
        if len(curr_child) != 0 and curr_tag[1] not in self.hard_code_label:
            prev_attr = curr_child[0].tag.split(self.SPT)[0]
            per_attr_seq = []
            for child_node in curr_child:
                curr_attr = child_node.tag.split(self.SPT)[0]
                if prev_attr != curr_attr:
                    if len(per_attr_seq) == 1:
                        curr_seq.extend(per_attr_seq)
                    else:
                        curr_seq.append(per_attr_seq)
                    per_attr_seq = []
                    prev_attr= curr_attr
                per_attr_seq = self.tree_to_list(ann_tree.subtree(child_node.identifier), per_attr_seq, label_to_id)
            if len(per_attr_seq) == 1 or len(per_attr_seq) == 0:
                curr_seq.extend(per_attr_seq)
            else:
                curr_seq.append(per_attr_seq)
        seq.extend(curr_seq)
        return seq

    def list_to_tree(self, ann_seq, root_tree, parent_id=None, attr="root", unlabel_to_token=False):
        temp_ann_seq = []
        if len(ann_seq) == 1:
            curr_tag = self.unlabel_ele(ann_seq[0]) if unlabel_to_token else ann_seq[0]
            curr_node = root_tree.create_node(str.join(self.SPT, [attr,curr_tag]), parent=parent_id)
            return root_tree 

        for ann in ann_seq:
            if type(ann) is list or self.has_child_node[ann]:
                temp_ann_seq.append(ann)
            else:
                temp_ann_seq.append([ann])
        ann_seq = temp_ann_seq

        if len(ann_seq):
            node_seq_list = self.div_by_node_list(ann_seq)
            for node_seq in node_seq_list:
                curr_tag = self.unlabel_ele(node_seq[0]) if unlabel_to_token else node_seq[0]
                curr_node = root_tree.create_node(str.join(self.SPT, [attr,curr_tag]), parent=parent_id)
                if hasattr(cst, curr_tag):
                    curr_node_attr_list = dir(getattr(cst, curr_tag))
                    attr_list = [attr for attr in curr_node_attr_list if attr in self.interest_attr_list]
                    assert(len(attr_list) == len(node_seq[1:]))
                    for attr_ele, attr_seq in zip(attr_list, node_seq[1:]):
                        root_tree = self.list_to_tree(attr_seq, root_tree, curr_node.identifier, attr_ele, unlabel_to_token)
        return root_tree

    def div_by_node_list(self, node_seq):
        node_list = []
        curr_node_list = []
        for id, ann_ele in enumerate(node_seq):
            if type(ann_ele) is not list and not hasattr(cst, ann_ele):
                curr_node_list.append(ann_ele)
                
            if type(ann_ele) is not list and not curr_node_list or type(ann_ele) is list:
                curr_node_list.append(ann_ele)
            else:
                node_list.append(curr_node_list)
        if len(curr_node_list):
            node_list.append(curr_node_list)
        return node_list

    def div_by_attr(self, ann_seq):
        attr_list = []
        attr_cnt = 0
        st = 0
        for id, ann_ele in enumerate(ann_seq):
            if ann_ele == "argst":
                attr_cnt += 1
            elif ann_ele == "argen":
                attr_cnt -= 1
            if attr_cnt == 0:
                attr_list.append(ann_seq[st:id+1])
                st = id+1
        return attr_list

    def div_by_node(self, node_seq):
        node_list = []
        node_cnt = 0
        st = 0
        for id, ann_ele in enumerate(node_seq):
            if ann_ele == "nodest":
                node_cnt += 1
            elif ann_ele == "nodeen":
                node_cnt -= 1
                if node_cnt == 0:
                    node_list.append(node_seq[st:id+2])
                    st = id+2
        return node_list

    def seq_to_tree(self, ann_seq, root_tree, parent_id=None, attr="root"):
        if len(ann_seq):
            node_seq_list = self.div_by_node(ann_seq)
            for node_seq in node_seq_list:
                curr_tag = node_seq[-1]
                curr_node = root_tree.create_node(str.join(self.SPT, [attr,curr_tag]), parent=parent_id)
                if hasattr(cst, curr_tag):
                    attr_list = [attr for attr in dir(getattr(cst, curr_tag)) if attr in self.interest_attr_list]
                    attr_seq_list = self.div_by_attr(node_seq[1:-2])
                    for attr_ele, attr_seq in zip(attr_list, attr_seq_list):
                        root_tree = self.seq_to_tree(attr_seq[1:-1], root_tree, curr_node.identifier, attr_ele)
        return root_tree

    def tree_to_cst(self, ann_tree, cst_node=None):
        curr_node = ann_tree.get_node(ann_tree.root)
        arg_name, class_name = curr_node.tag.split(self.SPT)

        if not hasattr(cst, class_name):
            return cst.parse_expression(class_name)

        curr_class = getattr(cst, class_name)


        if hasattr(typing, '_GenericAlias'):
            check_sequence = lambda x: hasattr(curr_class.__dict__['__annotations__'][x], '_name') and curr_class.__dict__['__annotations__'][x]._name == 'Sequence'
        else:
            check_sequence = lambda x: hasattr(curr_class.__dict__['__annotations__'][x], '_name') and type(curr_class.__dict__['__annotations__'][x]) == typing.Sequence

        arg_dict = {
         attr: [] if check_sequence(attr) else None
         for attr in dir(curr_class)
         if attr in self.interest_attr_list
        }

        for child_node in ann_tree.children(ann_tree.root):
            child_arg_name, child_class_name = child_node.tag.split(self.SPT)
            if check_sequence(child_arg_name):
                arg_dict[child_arg_name].append(self.tree_to_cst(ann_tree.subtree(child_node.identifier)))
            else:
                arg_dict[child_arg_name] = self.tree_to_cst(ann_tree.subtree(child_node.identifier))
        return curr_class(**arg_dict)

    def label_ele(self, ann_ele, ann_tree=None, debug=False):
        if ann_ele in self.hard_code_label:
            if ann_ele == "Attribute":
                ann_ele = str.join('.',[n.tag.split(self.SPT)[-1] for n in ann_tree.leaves()[::-1]])
            elif ann_ele == "Subscript":
                ann_ele = "{}[{}]".format(*[n.tag.split(self.SPT)[-1] for n in ann_tree.leaves()[::-1]])
            elif ann_ele == "Name":
                ann_ele = "{}".format(*[n.tag.split(self.SPT)[-1] for n in ann_tree.leaves()])
        try:
            return self.reverse_label_dict[ann_ele] if debug else ann_ele
        except Exception:
            raise Exception("Unknow type labeling. Call Junho park")

    def unlabel_ele(self, label_ele, problem_info=None):
        return self.label_dict[label_ele]

    def label_seq(self, encoded_seq, problem_info=None):
        return [self.label_ele(ann_ele, problem_info) for id, ann_ele in enumerate(encoded_seq)]

    def unlabel_seq(self, labeled_seq, problem_info=None):
        return [self.unlabel_ele(label_ele, problem_info) for label_ele in labeled_seq]

    def encode(self, source_path, problem_info=None, mode="list"):
        if os.path.isfile(source_path):
            with open(source_path) as body_file:
                body_file = body_file.read()
        else:
            body_file = source_path
            body_cst = cst.parse_module(body_file)

        essential_tree = Tree()
        essential_tree = self.cst_to_tree(body_cst, essential_tree, attr="root")

        encoded_seq = []
        if mode=="list":
            labeled_seq = self.tree_to_list(essential_tree, encoded_seq, label_to_id=(not self.debug))
        elif mode == "seq":
            encoded_seq = self.tree_to_seq(essential_tree, encoded_seq)
            labeled_seq = self.label_seq(encoded_seq, problem_info)

        return labeled_seq

    def decode(self, labeled_seq, problem_info=None, mode="list"):

        if mode=="list":
            recovered_tree = self.list_to_tree(labeled_seq, Tree(), unlabel_to_token=(not self.debug))
        elif mode=="seq":
            decoded_seq = self.unlabel_seq(labeled_seq, problem_info)
            recovered_tree = self.seq_to_tree(decoded_seq, Tree())

        recovered_cst = self.tree_to_cst(recovered_tree)
        recovered_cst = self.attach_gen_file(recovered_cst)
        recovered_module = cst.Module(body=self.attach_code(recovered_cst.body))

        generated_code = cst.Module([]).code_for_node(recovered_module)

        return generated_code

if __name__ == '__main__':
    from yacs.config import CfgNode as CN
    cfg = CN(new_allowed=True)
    cfg.header_path = 'test/src/header.py'
    cfg.footer_path = 'test/src/footer.py'
    cfg.var_range = 10
    cfg.const_range = 20
    cfg.SPT = '/'
    temp_converter = Converter(cfg)
    label_seq = temp_converter.encode(
     'test/src/body.py'
    )
    print(label_seq)
    generated_code = temp_converter.decode(label_seq)

    with open('test/out/gen.py', "w") as out_file:
        out_file.write(generated_code)
