from pyaichtools import Converter, DefaultCfg
import argparse

parser = argparse.ArgumentParser(description='Encode source code to label and Decode to source')
parser.add_argument('--ql_path', help='path to quality list file', default='test/src/ql.py')
parser.add_argument('--body_path', help='path to source code to test', default='test/src/body.py')
parser.add_argument('--output_path', help='path to output code which decoded', default='test/out/gen.py')

args = parser.parse_args()

# change here to test your answer code is right
body_path = args.body_path
output_path = args.output_path

DefaultCfg.ql_path = args.ql_path

temp_converter = Converter(DefaultCfg, debug=True)
label_seq = temp_converter.encode(body_path)
#label_seq = ['Module', ['SimpleStatementLine', ['Assign', ['AssignTarget', ['result']], ['BinaryOperation', ['QL[1]'], ['Multiply'], ['QL[1]']]]]]
generated_code = temp_converter.decode(label_seq)

print(generated_code)

"""
with open(output_path, "w") as out_file:
	out_file.write(generated_code)

exec(generated_code)
"""