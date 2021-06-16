# pyAICHtools : AI challenge source code converter (encode, decode)

## Usage:
```
tools/check_recovering.py --ql_path test/src/ql.py --body_path test/src/body.py --output_path test/out/gen.py
```

* ql_path	:	문제를 parsing하여 얻은 quality list python file 경로
* body_path	:	문제 답으로 작성한 code 
* output_path : Reconstructed 된 output code를 저장할 경로

## Install:

```
pip install -r requirements.txt
python setup.py build develop
```
