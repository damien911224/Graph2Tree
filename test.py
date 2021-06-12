from kobert_transformers import get_tokenizer
tokenizer = get_tokenizer()
a = tokenizer.tokenize("[CLS] 한국어 모델을 공유합니다. 10권의 책이 있습니다. [SEP]")
print(a)
b = tokenizer.convert_tokens_to_ids(a)
print(b)