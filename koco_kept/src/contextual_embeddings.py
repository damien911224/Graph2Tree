import torch.nn as nn
import torch
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, ElectraModel, ElectraTokenizer
from kobert_transformers import get_tokenizer
from transformers import ElectraModel, ElectraTokenizer
import konlpy
import pdb

def combine_num(all_tokens):
	num_index = []
	for sent in all_tokens:
		lenSent = len(sent)
		num_index_tmp = []
		for j in range(1, lenSent - 2):
			if j >= lenSent - 2:
				break
			if sent[j][-1] == 'N' and sent[j + 1][-1] == 'U' and sent[j + 2][-1] == 'M':
				sent[j] += 'UM'
				sent.pop(j + 1)
				sent.pop(j + 1)
				lenSent -= 2
				num_index_tmp.append(j)
		num_index.append(num_index_tmp)

	return all_tokens, num_index

def combine_num_(all_tokens):
	num_index = []
	for sent in all_tokens:
		lenSent = len(sent)
		num_index_tmp = []
		for j in range(1, lenSent - 2):
			if sent[j] == 'NUM':
				num_index_tmp.append(j)
		num_index.append(num_index_tmp)

	return all_tokens, num_index

class BertEncoder(nn.Module):
	def __init__(self, bert_model = 'kobert',device = 'cuda:0 ', freeze_bert = False):
		super(BertEncoder, self).__init__()
		if False:
			self.bert_layer = BertModel.from_pretrained(bert_model)
			self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
		else:
			if bert_model == None:
				bert_model = "monologg/koelectra-base-v3-discriminator"
			# self.bert_layer = BertModel.from_pretrained('monologg/kobert')
			# self.bert_tokenizer = get_tokenizer()
			self.bert_layer = ElectraModel.from_pretrained(bert_model)
			# original tokenizer
			# self.bert_tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
			# add names and "NUM" token
			self.bert_tokenizer = ElectraTokenizer.from_pretrained("./koelectra_tokenizer")

			# self.bert_tokenizer.save_pretrained("./tokenizer")

			# special_tokens_dict = {'additional_special_tokens': ["NUM", "정국", "지민", "석진", "태형", "남준", "윤기", "호석", "민영", "유정", "은지", "유나"]}
			# self.bert_tokenizer.add_special_tokens(special_tokens_dict)
			# self.bert_layer.resize_token_embeddings(len(self.bert_tokenizer))

			# tokens_dict = {'additional_tokens': ["NUM", "정국", "지민", "석진", "태형", "남준", "윤기", "호석", "민영", "유정", "은지", "유나"]}
			# self.bert_tokenizer.add_tokens(tokens_dict)
			# self.bert_layer.resize_token_embeddings(len(self.bert_tokenizer))
		self.device = device
		
		if freeze_bert:
			for p in self.bert_layer.parameters():
				p.requires_grad = False

	def bertify_input_(self, sentences):
		'''
		Preprocess the input sentences using bert tokenizer and converts them to a torch tensor containing token ids

		'''
		# Tokenize the input sentences for feeding into BERT
		# pdb.set_trace()
		# print(sentences[0])
		all_tokens = [['[CLS]'] + self.bert_tokenizer.tokenize(sentence) + ['[SEP]'] for sentence in sentences]

		_, num_index = combine_num_(all_tokens)
		# print(all_tokens[0])

		index_retrieve = num_index
		# print(index_retrieve)

		# Pad all the sentences to a maximum length
		input_lengths = [len(tokens) for tokens in all_tokens]
		max_length = max(input_lengths)
		padded_tokens = [tokens + ['[PAD]' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

		# Convert tokens to token ids
		token_ids = torch.tensor([self.bert_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(
			self.device)

		# Obtain attention masks
		pad_token = self.bert_tokenizer.convert_tokens_to_ids('[PAD]')
		attn_masks = (token_ids != pad_token).long()

		return token_ids, attn_masks, input_lengths, index_retrieve

	def bertify_input(self, sentences):
		'''
		Preprocess the input sentences using bert tokenizer and converts them to a torch tensor containing token ids

		'''
		#Tokenize the input sentences for feeding into BERT
		# pdb.set_trace()
		all_tokens  = [['[CLS]'] + self.bert_tokenizer.tokenize(sentence) + ['[SEP]'] for sentence in sentences]
		# print(all_tokens[0])
		# print(all_tokens[0])

		# TODO if you want to use in english case, have to change combine_num with undercase num and unk
		all_tokens, num_index = combine_num_(all_tokens)
		index_retrieve = num_index
		# print(all_tokens[0])
		# okt = konlpy.tag.Okt()
		# print(okt.morphs(sentences[0]))

		# index_retrieve = []
		# for sent in all_tokens:
		# 	cur_ls = []
		# 	for j in range(1, len(sent)):
		# 		if sent[j][0] == '#':
		# 			continue
		# 		else:
		# 			cur_ls.append(j)
		# 	index_retrieve.append(cur_ls)
		
		#Pad all the sentences to a maximum length
		input_lengths = [len(tokens) for tokens in all_tokens]
		max_length    = max(input_lengths)
		padded_tokens = [tokens + ['[PAD]' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

		#Convert tokens to token ids
		token_ids = torch.tensor([self.bert_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(self.device)

		#Obtain attention masks
		pad_token = self.bert_tokenizer.convert_tokens_to_ids('[PAD]')
		attn_masks = (token_ids != pad_token).long()

		return token_ids, attn_masks, input_lengths, index_retrieve

	def forward(self, sentences):
		'''
		Feed the batch of sentences to a BERT encoder to obtain contextualized representations of each token
		'''
		#Preprocess sentences
		token_ids, attn_masks, input_lengths, index_retrieve = self.bertify_input_(sentences)

		#Feed through bert
		# cont_reps, _ = self.bert_layer(token_ids, attention_mask = attn_masks)
		# Feed through ELECTRA
		cont_reps = self.bert_layer(token_ids, attention_mask = attn_masks)[0]

		return cont_reps, input_lengths, token_ids, index_retrieve

class RobertaEncoder(nn.Module):
	def __init__(self, roberta_model = 'roberta-base', device = 'cuda:0 ', freeze_roberta = False):
		super(RobertaEncoder, self).__init__()
		self.roberta_layer = RobertaModel.from_pretrained(roberta_model)
		self.roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_model)
		self.device = device
		
		if freeze_roberta:
			for p in self.roberta_layer.parameters():
				p.requires_grad = False
		
	def robertify_input(self, sentences):
		'''
		Preprocess the input sentences using roberta tokenizer and converts them to a torch tensor containing token ids

		'''
		# Tokenize the input sentences for feeding into RoBERTa
		all_tokens  = [['<s>'] + self.roberta_tokenizer.tokenize(sentence) + ['</s>'] for sentence in sentences]
		
		index_retrieve = []
		for sent in all_tokens:
			cur_ls = [1]
			for j in range(2, len(sent)):
				if sent[j][0] == '\u0120':
					cur_ls.append(j)
			index_retrieve.append(cur_ls)				
		
		# Pad all the sentences to a maximum length
		input_lengths = [len(tokens) for tokens in all_tokens]
		max_length    = max(input_lengths)
		padded_tokens = [tokens + ['<pad>' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

		# Convert tokens to token ids
		token_ids = torch.tensor([self.roberta_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(self.device)

		# Obtain attention masks
		pad_token = self.roberta_tokenizer.convert_tokens_to_ids('<pad>')
		attn_masks = (token_ids != pad_token).long()

		return token_ids, attn_masks, input_lengths, index_retrieve

	def forward(self, sentences):
		'''
		Feed the batch of sentences to a RoBERTa encoder to obtain contextualized representations of each token
		'''
		# Preprocess sentences
		token_ids, attn_masks, input_lengths, index_retrieve = self.robertify_input(sentences)

		# Feed through RoBERTa
		cont_reps, _ = self.roberta_layer(token_ids, attention_mask = attn_masks)

		return cont_reps, input_lengths, token_ids, index_retrieve