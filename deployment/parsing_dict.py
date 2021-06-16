#-*- coding:utf-8 -*-

numbers = [
    ("스물", 20),
    ("서른", 30),
    ("마흔", 40),
    ("쉰", 50),
    ("예순", 60),
    ("일흔", 70),
    ("여든", 80),
    ("아흔", 90),
    ("하나", 1),
    ("한", 1),
    ("두", 2),
    ("둘", 2),
    ("세", 3),
    ("셋", 3),
    ("네", 4),
    ("넷", 4),
    ("다섯", 5),
    ("여섯", 6),
    ("일곱", 7),
    ("여덟", 8),
    ("여덜", 8),
    ("아홉", 9),
    ("일", 1),
    ("이", 2),
    ("삼", 3),
    ("사", 4),
    ("오", 5),
    ("육", 6),
    ("칠", 7),
    ("팔", 8),
    ("구", 9),
    ("열", 10),
    ("십", 10),
    ("백", 100),
    ("천", 1000),
    ("만", 10000),
    ("억", 100000000),
    ("조", 1000000000000),
    ("경", 10000000000000000),
    ("해", 100000000000000000000),
]

number_types = {
    "키로": "kg",
    "키로그램": "kg",
    "킬로": "kg",
    "킬로그램": "kg",
    '킬로그람': "kg",
    "그램": "g",
    "그람": "g",
    "리터": "L",
    "밀리리터": "mL",
    "미리리터": "mL",
    "미리": "mL",
    "밀리": "mL",
    "센치미터": "cm",
    "센티미터": "cm",
    "밀리미터": "mm",
    "미터": "m",
    "개입": "개입",
    "개": "개",
    "명": "명",
    "원": "원",
    "묶음": "묶음",
    "단": "단",
    "모": "모",
    "세트": "세트",
    "병": "병",
    "장": "장",
    "박스": "박스",
    "봉지": "봉지",
    "팩": "팩",
    "줄": "줄",
    "망": "망",
    "포": "포",
    "말": "말",
    "캔": "캔",
    "판": "판",
    "자루": "자루",
    "가마니": "가마니",
    "통": "통",
    "다스": "다스",
    "권":"권",
    "쪽":"쪽"
}

float_nums = [
    ("일", 1),
    ("이", 2),
    ("삼", 3),
    ("사", 4),
    ("오", 5),
    ("육", 6),
    ("칠", 7),
    ("팔", 8),
    ("구", 9)
]
change_list1 = ['첫', '두', '세', '네', '다섯', '여섯', '일곱', '여덟', '아홉', '열', \
                '둘', '셋', '넷', \
                '한', \
                '스무', "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔" \
]
change_list2 = ["열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"]
change_list3 = ["한", '두', '세', '네', '다섯', '여섯', '일곱', '여덟', '아홉']
target_list1 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', \
            '2', '3', '4', \
            '1', \
            '20', '30', '40', '50', '60', '70', '80', '90'
]
target_list2 = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
target_list3 = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

change_list4 = ['일', '이', '삼', '사', '오', '육', '칠', '팔', '구']
change_list5 = ['이', '삼', '사', '오', '육', '칠', '팔', '구']
change_list6 = ['십', '백']
change_list7 = ['일', '이', '삼', '사', '오', '육', '칠', '팔', '구', '십', '백', '천', '만']
change_list8 = ['첫', '둘', '셋', '넷', '다섯', '여섯', '일곱', '여덟', '아홉']
unit=['kg','g','L','ml','mL','cm','mm','m','t','쪽','권','개입','개','명','원','묶음','단','모','세트','다스','병','장','박스','봉지','팩','줄','망','포','말','캔','판','자루','가마니','통']
target_list4 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
target_list5 = [10, 100, 1000, 10000]

NL_list = [
    '\(가\)', '\\(나\)', '\(다\)', '\(라\)', '\(마\)', '\(바\)', '\(사\)', '\(아\)', '\(자\)', '\(차\)', '\(카\)', '\(타\)', '\(파\)', '\(하\)', \
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'T', 'S', 'U', 'V', 'W', 'X', 'Y', 'Z', \
    '정국', '지민', '석진', '태형', '남준', '윤기', '호석', '민영', '유정', '은지', '유나', \
    '꼴찌', \
    '앞', '옆', '뒤', '가로', '세로', '왼쪽', '오른쪽', \
    '오전', '오후', \
    '월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일', \
    '아버지', '어머니', '할아버지', '할머니', '손자', '손녀', '조카', '이모', '삼촌', '동생', '누나', '오빠', \
    '손가락', '발가락', '팔', '다리', \
    '암컷', '수컷', '암탉', '수탉', '여학생', '남학생', '여차', '남자', \
    '흰색', '검은색', '파란색', '노란색', '초록색', '보라색', '빨간색', '주황색', '파란색', '남색', \
    '오리', '닭', '토끼', '물고기', '고래', '거위', '달팽이', '개구리', '강아지', '고양이', '비둘기', '병아리', '하마' \
    '연어', \
    '장미', '백합', '튤립', '카네이션', '국화', '화분', '화단', '꽃병', \
    '배구공', '농구공', '축구공', '탁구공', '줄넘기', '달리기', '수영', '시합', \
    '사과', '배', '감', '귤', '포도', '수박', '바나나', \
    '토마토', '무', '당근', '오이', '배추', \
    '나무', \
    '사탕', '김밥', '빵', '라면', '과자', '음료수', '주스', '우유', '달걀', '쌀', \
    '연필', '색연필', '지우개', '공책', '도화지', '색종이', '풀', '테이프', '바둑돌', '구슬', '상자', '나무토막', '장난감', '책장', '책꽂이', \
    '서점', '마트', '문구점', '집', '학교', '수영장', '교실', '도서관', '박물관', '운동장', '주차장', '정류장', '아파트', '농장', '강당', \
    '비행기', '자동차', '트럭', '배', '자전거', '오토바이', '기차', '버스', '엘리베이터', \
    '페인트', '벽', '천장', '문', '울타리'
]
name_list = ['정국이', '지민이', '석진이', '태형이', '남준이', '호석이', '민영이', '유정이']
josa_list = [
    '이', '가', '께서', '다', '을', '를', '의', '에' ,'로', '와', '과', '보다', '하고', '이랑', '랑', '이며', '며', '나', '만', '밖에', '뿐', '도', '조차', '마저', '까지', '이나', '나', \
    '이든지', '든지', '이나마', '나마', '이라도', '라도', '\s', '은', '는'
]