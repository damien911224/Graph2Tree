import stanza
import re
from nltk.tokenize import word_tokenize
from pororo import Pororo

nlp_stanza_eng = stanza.Pipeline(lang='en', processors='tokenize, pos, lemma, depparse')
# nlp_stanza_kor = stanza.Pipeline(lang='ko', processors='tokenize, pos, lemma, depparse')

def add_group_nums_eng(sent):
    sent = re.sub(r"-", r"", sent)
    sent = re.sub(r"mrs.", r"mrs", sent)
    sent_nums = re.findall('\d*\.?\d+', sent)
    doc = nlp_stanza_eng(sent)
    sent = word_tokenize(sent)

    final_ids = []
    assoc_nouns = []
    adjectives = []
    assoc_verbs = []
    rates = []

    offset = 0

    for s in doc.sentences:
        last_id = 0
        for word in s.words:
            if word.text in sent_nums:
                final_ids.append(offset + word.id - 1)
                if offset + (word.id - 1) - 1 >= 0 and sent[offset + (word.id - 1) - 1] not in [',', '.', ';']:
                    final_ids.append(offset + (word.id - 1) - 1)
                if offset + (word.id - 1) + 1 < len(sent) and sent[offset + (word.id - 1) + 1] not in [',', '.', ';']:
                    final_ids.append(offset + (word.id - 1) + 1)
                if word.deprel in ['nummod', 'nmode']:
                    assoc_nouns.append(s.words[word.head - 1].text)
                    final_ids.append(offset + word.head - 1)
            if word.text in ['each', 'every', 'per']:
                rates.append(word.text)
                final_ids.append(offset + word.id - 1)
            last_id = word.id
        offset += last_id

    offset = 0

    for s in doc.sentences:
        last_id = 0
        for word in s.words:
            if word.deprel == 'amod':
                if s.words[word.head - 1].text in assoc_nouns:
                    adjectives.append(word.text)
                    final_ids.append(word.id - 1)
            if word.text in assoc_nouns and word.deprel in ['obj', 'nsubj']:
                assoc_verbs.append(s.words[word.head - 1].text)
                final_ids.append(word.head - 1)
            last_id = word.id
        offset += last_id

    if len(sent) - 4 >= 0 and sent[len(sent) - 4] not in [',', '.', ';']:
        final_ids.append(len(sent) - 4)
    if len(sent) - 3 >= 0 and sent[len(sent) - 3] not in [',', '.', ';']:
        final_ids.append(len(sent) - 3)
    if len(sent) - 2 >= 0 and sent[len(sent) - 2] not in [',', '.', ';']:
        final_ids.append(len(sent) - 2)

    return list(set(final_ids))

def add_group_nums_kor(sent):
    sent = re.sub(r"-", r"", sent)
    sent = re.sub(r"mrs.", r"mrs", sent)
    sent_nums = re.findall('\d*\.?\d+', sent)
    doc = nlp_stanza_kor(sent)
    # sent = word_tokenize(sent)
    tk = Pororo(task="tokenization", lang="ko", model="bpe32k.ko", )
    sent = tk(sent)
    sent = [w.replace("_", "") for w in sent]

    final_ids = []
    assoc_nouns = []
    adjectives = []
    assoc_verbs = []
    rates = []

    offset = 0

    for s in doc.sentences:
        last_id = 0
        for word in s.words:
            if word.text in sent_nums:
                final_ids.append(offset + word.id - 1)
                if offset + (word.id - 1) - 1 >= 0 and sent[offset + (word.id - 1) - 1] not in [',', '.', ';']:
                    final_ids.append(offset + (word.id - 1) - 1)
                if offset + (word.id - 1) + 1 < len(sent) and sent[offset + (word.id - 1) + 1] not in [',', '.', ';']:
                    final_ids.append(offset + (word.id - 1) + 1)
                if word.deprel in ['nummod', 'nmode']:
                    assoc_nouns.append(s.words[word.head - 1].text)
                    final_ids.append(offset + word.head - 1)
            if word.text in ['each', 'every', 'per']:
                rates.append(word.text)
                final_ids.append(offset + word.id - 1)
            last_id = word.id
        offset += last_id

    offset = 0

    for s in doc.sentences:
        last_id = 0
        for word in s.words:
            if word.deprel == 'amod':
                if s.words[word.head - 1].text in assoc_nouns:
                    adjectives.append(word.text)
                    final_ids.append(word.id - 1)
            if word.text in assoc_nouns and word.deprel in ['obj', 'nsubj']:
                assoc_verbs.append(s.words[word.head - 1].text)
                final_ids.append(word.head - 1)
            last_id = word.id
        offset += last_id

    if len(sent) - 4 >= 0 and sent[len(sent) - 4] not in [',', '.', ';']:
        final_ids.append(len(sent) - 4)
    if len(sent) - 3 >= 0 and sent[len(sent) - 3] not in [',', '.', ';']:
        final_ids.append(len(sent) - 3)
    if len(sent) - 2 >= 0 and sent[len(sent) - 2] not in [',', '.', ';']:
        final_ids.append(len(sent) - 2)

    return list(set(final_ids))

if __name__ == "__main__":

    eng_sentence = "Bryan took a look at his books as well . If Bryan has 56 books in each of his 9 bookshelves , how many books does he have in total ?"
    kor_sentence = "브라이언은 또한 그의 책들을 살펴보았다. 브라이언이 그의 9개의 책꽂이 각각 56권의 책을 가지고 있다면, 그는 총 몇권의 책을 가지고 있는가?"

    eng_result = add_group_nums_eng(eng_sentence)
    kor_result = add_group_nums_kor(kor_sentence)

    print(eng_result)
    print(kor_result)