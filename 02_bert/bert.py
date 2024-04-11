import re
import math
import torch
import numpy as np
from random import *
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

'''
reference: https://wmathor.com/index.php/archives/1457/
'''
text = (
    'Hello, how are you? I am Romeo.\n'  # R
    'Hello, Romeo My name is Juliet. Nice to meet you.\n'  # J
    'Nice meet you too. How are you today?\n'  # R
    'Great. My baseball team won the competition.\n'  # J
    'Oh Congratulations, Juliet\n'  # R
    'Thank you Romeo\n'  # J
    'Where are you going today?\n'  # R
    'I am going shopping. What about you?\n'  # J
    'I am going to visit my grandmother. she is not very well'  # R
)
sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')  # filter '.', ',', '?', '!'
word_list = list(set(" ".join(sentences).split()))  # ['hello', 'how', 'are', 'you',...]
word2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
for i, w in enumerate(word_list):
    word2idx[w] = i + 4
idx2word = {i: w for i, w in enumerate(word2idx)}
vocab_size = len(word2idx)

token_list = list()
for sentence in sentences:
    arr = [word2idx[s] for s in sentence.split()]
    token_list.append(arr)

# BERT Parameters
maxlen = 30  # 表示同一个 batch 中的所有句子都由 30 个 token 组成，不够的补 PAD
batch_size = 6
max_pred = 5  # max tokens of prediction
n_layers = 6  # 表示 Encoder Layer 的数量
n_heads = 12
d_model = 768
d_ff = 768 * 4  # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2  # 表示 Decoder input 由几句话组成


# sample IsNext and NotNext to be same in small batch size
def make_data():
    batch = []
    positive = negative = 0  # positive 变量代表两句话是连续的个数，negative 代表两句话不是连续的个数，我们需要做到在一个 batch 中，这两个样本的比例为 1:1。
    while positive != batch_size / 2 or negative != batch_size / 2:
        tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(
            len(sentences))  # sample random index in sentences
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # MASK LM
        n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15)))  # 15 % of tokens in one sentence
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word2idx['[CLS]'] and token != word2idx['[SEP]']]  # candidate masked position
        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:  # 80%
                input_ids[pos] = word2idx['[MASK]']  # make mask
            elif random() > 0.9:  # 10%
                index = randint(0, vocab_size - 1)  # random index in vocabulary
                while index < 4:  # can't involve 'CLS', 'SEP', 'PAD'
                    index = randint(0, vocab_size - 1)
                input_ids[pos] = index  # replace

        # Zero Paddings
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # Zero Padding (100% - 15%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)  # 补齐句子的长度，使得一个 batch 中的句子都是相同长度。
            masked_pos.extend([0] * n_pad)  # 补齐 mask 的数量, 保证同一个 batch 中，mask 的数量（必须）是相同的

        # 随机选取的两句话是否连续，只要通过判断 tokens_a_index + 1 == tokens_b_index 即可
        if tokens_a_index + 1 == tokens_b_index and positive < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])  # IsNext
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])  # NotNext
            negative += 1
    return batch


# Proprecessing Finished
batch = make_data()
input_ids, segment_ids, masked_tokens, masked_pos, isNext = zip(*batch)
input_ids, segment_ids, masked_tokens, masked_pos, isNext = \
    torch.LongTensor(input_ids), torch.LongTensor(segment_ids), torch.LongTensor(masked_tokens), \
        torch.LongTensor(masked_pos), torch.LongTensor(isNext)


class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, segment_ids, masked_tokens, masked_pos, isNext):
        self.input_ids = input_ids             # 输入
        self.segment_ids = segment_ids        # 区分不同句子或不同段落
        self.masked_tokens = masked_tokens    # 为进行Masked Language Model (MLM) 的预训练，会将一部分token进行mask操作，模型需要预测这些被mask的token。
        self.masked_pos = masked_pos          # 在BERT中，为了进行MLM的预训练，需要知道哪些位置的token被mask了。 （可学习）
        self.isNext = isNext                  # 是否连续，True为连续，False为不连续

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.segment_ids[idx], self.masked_tokens[idx], self.masked_pos[idx], self.isNext[
            idx]


loader = Data.DataLoader(MyDataSet(input_ids, segment_ids, masked_tokens, masked_pos, isNext), batch_size, True)

print(loader)

