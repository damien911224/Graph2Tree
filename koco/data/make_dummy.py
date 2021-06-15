import json

with open("custom.json", "r") as fp:
    data = json.load(fp)

dummy_code = ["Module", ["For", ["IndentedBlock", ["If", ["IndentedBlock", ["SimpleStatementLine", ["AugAssign", ["AddAssign"], ["var0"], ["BinaryOperation", ["Call", ["Arg", ["const1"]], ["math.acos"]], ["Add"], ["Call", ["Arg", ["var2"]], ["math.atan"]]]]]], ["Comparison", ["ComparisonTarget", ["QL[2]"], ["GreaterThan"]], ["var1"]]]], ["Call", ["Arg", ["List", ["Element", ["QL[0]"], "Element", ["QL[1]"], "Element", ["QL[2]"], "Element", ["QL[3]"], "Element", ["QL[4]"]]], "Arg", ["const2"]], ["itertools.combinations"]], ["Tuple", ["Element", ["var1"], "Element", ["var2"]]], "SimpleStatementLine", ["Assign", ["AssignTarget", ["result"]], ["var0"]]]] 
for datum in data:
    datum["lEquations"] = dummy_code

with open("custom_dummy.json", "w") as fp:
    json.dump(data, fp, indent=4)