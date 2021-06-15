import json

with open("mawps_combine.json", "r") as fp:
    data = json.load(fp)

dummy_code = ['Module', ['For', ['IndentedBlock', ['If', ['IndentedBlock', ['SimpleStatementLine', ['AugAssign', ['AddAssign'], ['Name', ['var0']], ['BinaryOperation', ['Call', ['Arg', ['Name', ['const1']]], ['Attribute', ['Name', ['acos']], ['Name', ['math']]]], ['Add'], ['Call', ['Arg', ['Name', ['var2']]], ['Attribute', ['Name', ['atan']], ['Name', ['math']]]]]]]], ['Comparison', ['ComparisonTarget', ['Subscript', ['SubscriptElement', ['Index', ['Integer', ['2']]]], ['Name', ['QL']]], ['GreaterThan']], ['Name', ['var1']]]]], ['Call', ['Arg', ['List', ['Element', ['Subscript', ['SubscriptElement', ['Index', ['Integer', ['0']]]], ['Name', ['QL']]], 'Element', ['Subscript', ['SubscriptElement', ['Index', ['Integer', ['1']]]], ['Name', ['QL']]], 'Element', ['Subscript', ['SubscriptElement', ['Index', ['Integer', ['2']]]], ['Name', ['QL']]], 'Element', ['Subscript', ['SubscriptElement', ['Index', ['Integer', ['3']]]], ['Name', ['QL']]], 'Element', ['Subscript', ['SubscriptElement', ['Index', ['Integer', ['4']]]], ['Name', ['QL']]]]], 'Arg', ['Name', ['const2']]], ['Attribute', ['Name', ['combinations']], ['Name', ['itertools']]]], ['Tuple', ['Element', ['Name', ['var1']], 'Element', ['Name', ['var2']]]], 'SimpleStatementLine', ['Assign', ['AssignTarget', ['Name', ['result']]], ['Name', ['var0']]]]]
for datum in data:
    datum["lEquations"] = dummy_code

with open("mawps_dummy.json", "w") as fp:
    json.dump(data, fp, indent=4)