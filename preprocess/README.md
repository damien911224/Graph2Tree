## QLNL 추출기

### install.sh
* konlp에 관련된 라이브러리를 설치합니다.
* open jdk 1.8을 설치하는데, 이미 깔려있으면 알아서 수정해서 설치하셔야 합니다.
실행방법:
~~~
sh install.sh
~~~

### main.py
* 각자 제시한 문제지 파일 이름을 --input 인자로 입력합니다. 입력하지 않으면 디폴트는 'problem.json' 입니다.
* python 3.7에서만 테스트를 했습니다.
* main.py를 실행시키면 problem.json 과 answer.json을 취합하여 데이터를 얻을 수 있습니다.
* --prob 옵션과 --ans 옵션은 필수이며, 각각 파일이름을 지정해야 합니다. --data 옵션은 출력 데이트를 지정하는 경로를 지정합니다. (default : data/output.json)
~~~
python --prob problems_final.json --ans answers_final.json
python --prob problems_final.json --ans answers_final.json --data data/output.json # 출력 파일 경로 지정
~~~

### 결과 파일 종류

1. error_list.txt : 문제와 정답파일에서 데이터를 추출할때 에러가 발생하는 경우 이곳에 모든 에러를 저장합니다.
2. diff.txt : 데이터셋 QL과 추출기 QL이 다를 경우 문제번호를 출력합니다.
3. data/output.json
