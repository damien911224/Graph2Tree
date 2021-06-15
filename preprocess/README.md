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
* main.py를 실행시키면 qlnl_list.txt 파일로 결과가 출력됩니다.
~~~
python main.py # default
python main.py --input problem.json # change input file name
~~~
