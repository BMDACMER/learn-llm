

二级标题

## LLM



### Transformer架构



<center class="half">
    <img src="./asset/transformer.png" width="650" height="520"/>
    <img src="./asset/transformer2.png" width="450" height="400"/>
</center>

原始论文中的N=6

```
# Transformer Parameters
d_model = 512  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention
```





#### 嵌入层Position Encoding

ransformer 中除了单词的 Embedding，还需要使用位置 Embedding 表示单词出现在句子中的位置。**因为 Transformer 不采用 RNN 的结构，而是使用全局信息，不能利用单词的顺序信息。** 为了维持句子之间的关系，transformer使用 position encoding来保持单词在序列中相对位置或绝对位置。

位置 Embedding 用 **PE**表示，**PE** 的维度与单词 Embedding 是一样的。PE 可以通过训练得到，也可以使用某种公式计算得到。在 Transformer 中采用了后者，计算公式如下：

![img](E:\paper\LLM\00-LLM个人学习笔记20240409\asset\位置编码.jpg)

> 可以让模型容易地计算出相对位置，对于固定长度的间距 k，**PE(pos+k)** 可以用 **PE(pos)** 计算得到。因为 Sin(A+B) = Sin(A)Cos(B) + Cos(A)Sin(B), Cos(A+B) = Cos(A)Cos(B) - Sin(A)Sin(B)。

将单词的词 Embedding 和位置 Embedding 相加，就可以得到单词的表示向量 **x**，**x** 就是 Transformer 的输入。

**手撕代码**

```python
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))  # (d_model/2, )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)  # [max_len, d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]  # 后面的两维与x为准
        return self.dropout(x)

```

==绝对位置编码是对数据进行修饰，相对位置编码是对注意力得分的A矩阵修饰==



#### 注意力Attention原理和实现

##### self-Attention 机制

![self-attention](E:\paper\LLM\00-LLM个人学习笔记20240409\asset\self-attention.png)

##### Q,K,V的计算

<img src="E:\paper\LLM\00-LLM个人学习笔记20240409\asset\attention01.png" style="zoom:50%;" />

Self-Attention 的输入用矩阵X进行表示，则可以使用线性变阵矩阵**WQ,WK,WV**计算得到**Q,K,V**。计算如下图所示，**注意 X, Q, K, V 的每一行都表示一个单词。**

接着通过下面公式计算self-attention输出：

<img src="E:\paper\LLM\00-LLM个人学习笔记20240409\asset\attention02.png" style="zoom:67%;" />

下面是计算Q,K,V流程示意图。

<center class="half">
    <img src="./asset/attention03.png" width="520" height="200"/>
    <img src="./asset/attention04.png" width="520" height="200"/>
    <img src="./asset/attention05.png" width="520" height="200"/>
    <img src="./asset/attention06.png" width="520" height="200"/>
</center>





手撕代码如下：

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn
```

多头注意力机制：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).cuda()(output + residual), attn
```

完整代码中一定会有三处地方调用 `MultiHeadAttention()`，Encoder Layer 调用一次，传入的 `input_Q`、`input_K`、`input_V` 全部都是 `enc_inputs`；Decoder Layer 中两次调用，第一次传入的全是 `dec_inputs`，第二次传入的分别是 `dec_outputs`，`enc_outputs`，`enc_outputs`



#### Encoder-Decoder 实现

##### Encoder

首先定义**encoder layer**，将上述模块结合起来，代码如下：

```python
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()      # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet()         #   Feed Forward 前馈网络（也即MPLs）

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn
```

**Encoder**模块：

```python
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)      # 对输入src_vocab 做embedding
        self.pos_emb = PositionalEncoding(d_model)                # 位置编码
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])   # 堆叠n层 EncoderLayer

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)  # 由于我们控制好了 Encoder Layer 的输入和输出维度相同，所以可以直接用个 for 循环以嵌套的方式，将上一次 Encoder Layer 的输出作为下一次 Encoder Layer 的输入
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns
```



##### Decoder

Decoder 层

```python
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()      #  decoder自注意力层
        self.dec_enc_attn = MultiHeadAttention()       # decoder经过上一层多头注意力层输出Q 和 来自encoder输出k，v 一同输入到 多头注意力层
        self.pos_ffn = PoswiseFeedForwardNet()         # 前馈神经网络

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs) # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn
```

**Decoder**模块

```python
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)    #  处理decoder输入
        self.pos_emb = PositionalEncoding(d_model)              # 位置编码
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])   # 堆叠n层decoderlayer

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]
        '''
        dec_outputs = self.tgt_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).cuda() # [batch_size, tgt_len, d_model]
        # Decoder 中不仅要把 "pad"mask 掉，还要 mask 未来时刻的信息，因此就有了下面这三行代码，其中 torch.gt(a, value) 的意思是，将 a 中各个位置上的元素和 value 比较，若大于 value，则该位置取 1，否则取 0
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda() # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda() # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0).cuda() # [batch_size, tgt_len, tgt_len]

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs) # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns
```





#### 详解Masked原理

mask主要有两处：一处是pad mask 另一处是Sequence Mask。

- Pad Mask的作用是处理变长序列，为了输入对齐，确保模型不会再填充部分计算。
- Sequence Mask 的目的是 在自注意力机制中，确保每个位置只能与==自身之前的位置==进行注意力计算，而不与自身之后的位置进行注意力计算。



**Pad Mask**

```python
def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]
```

这个函数最核心的一句代码是 `seq_k.data.eq(0)`，这句的作用是返回一个大小和 `seq_k` 一样的 tensor，只不过里面的值只有 True 和 False。如果 `seq_k` 某个位置的值等于 0，那么对应位置就是 True，否则即为 False。举个例子，输入为 `seq_data = [1, 2, 3, 4, 0]`，`seq_data.data.eq(0)` 就会返回 `[False, False, False, False, True]`

**Sequence Mask**

```python
def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask # [batch_size, tgt_len, tgt_len]
```

Subsequence Mask 只有 Decoder 会用到，主要作用是屏蔽未来时刻单词的信息。首先通过 `np.ones()` 生成一个全 1 的方阵，然后通过 `np.triu()` 生成一个上三角矩阵，下图是 `np.triu()` 用法

<img src="E:\paper\LLM\00-LLM个人学习笔记20240409\asset\sequence-mask.png" style="zoom:80%;" />



#### Transformer

```python
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().cuda()                                           # Encoder
        self.decoder = Decoder().cuda()                                           # Decoder
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()   # 线性层  做空间变换
    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs) # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
```

Transformer 主要就是调用 Encoder 和 Decoder。最后返回 `dec_logits` 的维度是 [batch_size * tgt_len, tgt_vocab_size]，可以理解为，一个句子，这个句子有 batch_size*tgt_len 个单词，每个单词有 tgt_vocab_size 种情况，取概率最大者。



#### 其他

矩阵相乘相当于空间变换，原来向量有的性质在经过空间变换后保持不变。矩阵的行代表的是旧坐标系有多少个维度，矩阵的列代表的是新坐标系有多少维度。绝对位置编码是对数据进行修饰，相对位置编码是对注意力得分的A矩阵修饰。隐藏层越深抽象程度越高。怎么操作数据与数据本身都无关，只与相乘的矩阵有关（做空间变化）。



**参考资料**：[transformer原理讲解和代码](https://wmathor.com/index.php/archives/1455/)

[图片参考](https://zhuanlan.zhihu.com/p/338817680)





### Bert

**问题1：bert的具体网络结构，以及训练过程，bert为什么火，它在什么的基础上改进了些什么？**

>  bert是用了transformer的encoder侧的网络，作为一个文本编码器，使用大规模数据进行预训练，预训练使用两个loss，一个是mask LM，遮蔽掉源端的一些字（可能会被问到mask的具体做法，15%概率mask词，这其中80%用[mask]替换，10%随机替换一个其他字，10%不替换，至于为什么这么做，那就得问问BERT的作者了），然后根据上下文去预测这些字，一个是next sentence，判断两个句子是否在文章中互为上下句，然后使用了大规模的语料去预训练。在它之前是GPT，GPT是一个单向语言模型的预训练过程（**它和gpt的区别就是bert为啥叫双向 bi-directional**），更适用于文本生成，通过前文去预测当前的字。
>
> BERT（Bidirectional Encoder Representations from Transformers）之所以被称为双向（bidirectional），是因为它在预训练阶段考虑了文本序列的双向上下文信息。==BERT利用了Transformer模型的自注意力机制==，使得它在预训练阶段能够同时考虑到当前词之前和之后的词的信息，从而获得了双向的上下文信息。这样的设计使得BERT在理解文本时能够更好地把握整个句子的语境，提高了模型的性能和泛化能力。



**问题2，讲讲multi-head attention的具体结构**

BERT由12层transformer layer（encoder端）构成，首先word emb , pos emb（可能会被问到有哪几种position embedding的方式，bert是使用的哪种）, sent emb做加和作为网络输入，每层由一个multi-head attention, 一个feed forward 以及两层layerNorm构成，一般会被问到multi-head attention的结构，具体可以描述为，

**step1**

一个768的hidden向量，被映射成query， key， value。 然后三个向量分别切分成12个小的64维的向量，每一组小向量之间做attention。不妨假设batch_size为32，seqlen为512，隐层维度为768，12个head

hidden(32 x 512 x 768) -> query(32 x 512 x 768) -> 32 x 12 x 512 x 64

hidden(32 x 512 x 768) -> key(32 x 512 x 768) -> 32 x 12 x 512 x 64

hidden(32 x 512 x 768) -> val(32 x 512 x 768) -> 32 x 12 x 512 x 64

**step2**

然后query和key之间做attention，得到一个32 x 12 x 512 x 512的权重矩阵，然后根据这个权重矩阵加权value中切分好的向量，得到一个32 x 12 x 512 x 64 的向量，拉平输出为768向量。

32 x 12 x 512 x 64(query_hidden) * 32 x 12 x 64 x 512(key_hidden) -> 32 x 12 x 512 x 512

32 x 12 x 64 x 512(value_hidden) * 32 x 12 x 512 x 512 (权重矩阵) -> 32 x 12 x 512 x 64

然后再还原成 -> 32 x 512 x 768

简言之是12个头，每个头都是一个64维度分别去与其他的所有位置的hidden embedding做attention然后再合并还原。

**代码如下**：

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size, seq_len, n_heads * d_v]
        output = nn.Linear(n_heads * d_v, d_model)(context)
        return nn.LayerNorm(d_model)(output + residual) # output: [batch_size, seq_len, d_model]
```



**问题2.5: Bert 采用哪种Normalization结构，LayerNorm和BatchNorm区别，LayerNorm结构有参数吗，参数的作用？**

> 采用LayerNorm结构，和BatchNorm的区别主要是做规范化的维度不同，BatchNorm针对一个batch里面的数据进行规范化，针对单个神经元进行，比如batch里面有64个样本，那么规范化输入的这64个样本各自经过这个神经元后的值（64维），LayerNorm则是针对单个样本，不依赖于其他数据，常被用于小mini-batch场景、动态网络场景和 RNN，特别是自然语言处理领域，就bert来说就是对每层输出的隐层向量（768维）做规范化，图像领域用BN比较多的原因是因为每一个卷积核的参数在不同位置的神经元当中是共享的，因此也应该被一起规范化。

```python
class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
```

Tip: 归一化不是就是把上层的输出约束为一个正态分布，为什么还有个w和b的参数？

> w，b这两个参数归一化是为了让本层网络的输出进行额外的约束，但如果每个网络的输出被限制在这个约束内，就会限制模型的表达能力。 通俗的讲，不管你上层网络输出的是什么牛鬼蛇神的向量，都先给我归一化到正态分布（均值为0，标准差为1），但我不能给下游传这个向量，如果这样下游网络看到的只能是归一化后的，限制了模型。



**问题3：transformer attention的时候为啥要除以根号D**

> 参考 transformer面经



**问题4：wordpiece的作用**

> **WordPiece模型的核心思想**：将单词拆分为字符，并根据字符片段的组合频率来动态地划分单词。这意味着模型可以通过组合不同的字符片段来构建词汇表，并根据语料库中的频率来动态地调整词汇表的大小和内容。
>
> **降低OOV**: 相比于传统的分词方法，WordPiece模型能够极大地降低Out of Vocabulary（OOV）的情况。举例来说，对于像"cosplayer"这样的罕见词，传统的分词方法可能会将其标记为UNK（未知词），而WordPiece模型可以将其拆分为更小的字符片段，比如"cos play er"，这样模型就能够学习到词的词根、前缀等信息，而不是将其视为一个未知词。
>
> **学习到更多信息**：由于WordPiece模型将单词拆分为字符片段，并根据片段的组合频率来动态地划分单词，因此模型能够更好地学习到单词的内部结构和语义信息。这样可以提高模型对于罕见词和词汇表之外词的处理能力，增强了模型的泛化能力和性能。



**问题5： 如何优化BERT效果：**

> 1、感觉最有效的方式还是数据。
>
> 2、BERT上面加一些网络结构，比如attention，rcnn等，个人得到的结果感觉和直接在上面加一层transformer layer的效果差不多，模型更加复杂，效果略好，计算时间略增加。
>
> 3、改进预训练，在特定的大规模数据上预训练，相比于开源的用百科，知道等数据训练的更适合你的任务（经过多方验证是一种比较有效的提升方案）。以及在预训练的时候去mask低频词或者实体词（听说过有人这么做有收益，但没具体验证）。



**问题6 如何优化BERT性能**

> 1 压缩层数，然后蒸馏，直接复用12层bert的前4层或者前6层，效果能和12层基本持平，如果不蒸馏会差一些。
>
> 2 双塔模型（短文本匹配任务），将bert作为一个encoder，输入query编码成向量，输入title编码成向量，最后加一个DNN网络计算打分即可。离线缓存编码后的向量，在线计算只需要计算DNN网络。
>
> 3 int8预估，在保证模型精度的前提下，将Float32的模型转换成Int8的模型。
>
> 4 提前结束，大致思想是简单的case前面几层就可以输出分类结果，比较难区分的case走完12层，但这个在batch里面计算应该怎么优化还没看明白，有的提前结束有的最后结束，如果在一个batch里面的话就不太好弄。



**问题7 self-attention相比lstm优点是什么？**

> bert通过使用self-attention + position embedding对序列进行编码，lstm的计算过程是从左到右从上到下（如果是多层lstm的话），后一个时间节点的emb需要等前面的算完，而==bert这种方式相当于并行计算，虽然模型复杂了很多，速度其实差不多==。



### 面试题

#### Transformer

1. Transformer为何使用多头注意力机制？（为什么不使用一个头）

   > 主要原因是为了增加模型的表达能力和捕捉输入序列中不同特征之间的关系。--- 多视角学习
   >
   > 其次，使用多头注意力机制可以将注意力计算拆分成多个头，每个头可以并行计算，从而提高了计算效率。--- 并行性

2. Transformer为什么Q和K使用不同的权重矩阵生成，为何不能使用同一个值进行自身的点乘？ （注意和第一个问题的区别）

   > 主要原因是为了增加模型的灵活性和表达能力。虽然理论上可以使用相同的权重矩阵进行自身的点乘，但是使用不同的权重矩阵可以让模型学习到更加丰富和复杂的信息。 以下是使用不同权重的优势：
   >
   > 1）模型泛化能力：通过使用不同的权重矩阵，模型可以学习到不同的表示，从而更好地捕捉输入序列中的不同特征。如果Q和K使用相同的权重矩阵，那么在计算点积时，每个输入向量都会产生相同的注意力分数，这将导致模型无法区分不同的输入向量。
   >
   > 2）灵活性：通过使用不同的权重矩阵，模型可以分别学习到如何根据查询向量的需求来筛选和组合键向量中的信息。
   >
   > 3）可解释性：使用不同的权重矩阵有助于提高模型的可解释性。当模型使用不同的权重矩阵时，我们可以更容易地分析和理解模型是如何根据查询向量的需求来选择和组合键向量中的信息的。这有助于我们深入了解模型的工作原理和行为。

3. Transformer计算attention的时候为何选择点乘而不是加法？两者计算复杂度和效果上有什么区别？

   > 1. **线性变换的表达能力**： 点乘是一种线性变换，它可以捕捉输入向量之间的相似性。在自注意力机制中，点乘用于计算查询（Query）和键（Key）之间的相似度。如果两个向量的点乘结果较大，表示它们之间的相似度高，相应的值（Value）在输出中会被赋予较高的权重。而加法是一种非线性运算，它不具备捕捉这种相似性的能力。
   > 2. **缩放和标准化的特性**： 点乘后通常会进行缩放操作（例如，除以键向量的维度的平方根），这有助于避免梯度消失或爆炸问题，并使得注意力分数更加稳定。此外，点乘可以自然地与softmax函数结合，进行归一化处理，使得每个位置的注意力分数总和为1，这有助于模型聚焦于最重要的信息。而加法不具有这样的特性。
   > 3. **计算复杂度**： 尽管点乘和加法在计算上都是线性的，但点乘在自注意力机制中的应用可以有效地并行化，因为每个查询和键之间的点乘可以独立计算。相比之下，如果使用加法，可能需要更多的步骤来确保结果的正确性，例如需要先进行排序或其他操作。
   > 4. **模型的效果**： 在实践中，点乘已经被证明是计算注意力分数的有效方法。它不仅可以捕捉到序列中不同部分之间的复杂关系，还可以通过注意力机制将这些关系转化为最终的输出。而加法可能无法捕捉到这些细微的关系，因为它不具备点乘那样的区分度。
   > 5. **可扩展性**： 点乘操作可以很容易地扩展到多维空间，这使得模型可以处理更高维度的特征。而加法则可能在高维空间中遇到“维度的诅咒”，导致效果不佳。

4. 为什么在进行softmax之前需要对attention进行scaled（为什么除以dk的平方根），并使用公式推导进行讲解

   > 在进行softmax之前对注意力进行缩放（scaled）的目的是为了==控制注意力权重的大小==，以避免在 softmax 函数中出现==数值不稳定==的情况，并且可以确保梯度不会随着输入维度的增加而变得过小。
   >
   > 为什么选择除以$\sqrt{d_k}$作为缩放因子呢？这可以从几何和概率的角度来理解。假设我们有一个高斯分布（正态分布），其均值为0，方差为1（即标准正态分布）。在这个分布中，一个随机变量落在±$\delta$范围内的概率约为68.27%，其中$\delta$是标准差。如果我们将这个分布推广到多维空间，一个点乘操作可以看作是在计算两个向量之间的相似度，其结果的分布的方差应该是$d_k$（因为==每个维度都是独立的==）。为了使得相似度的分布保持在合理的范围内，我们需要将方差缩放到1，即$\frac{1}{\sqrt{d_k}}$。这样，点乘结果的标准差就是1，保证了分布的稳定性。
   >
   > Scaled dot-product attention(*Q*,*K*,*V*)=$sofrmax(\frac{QK^T}{\sqrt{d_k}})V$

5. 在计算attention score的时候如何对padding做mask操作？

   > 在计算注意力分数时，对填充进行掩码操作是为了在计算注意力权重时忽略填充位置，以确保模型不会在填充部分计算注意力。 具体做法如下：
   >
   > - 将填充位置的值设置为一个很大的负数或者特别小的数，比如 -1e9。
   >   - *# 将mask矩阵扩展到与注意力分数相同的维度* attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
   > - 在计算注意力分数时，将填充位置的值替换为上述的负数值。
   > - 然后应用 softmax 函数，得到注意力权重

6. 为什么在进行多头注意力的时候需要对每个head进行降维？

   > 1. **降低计算复杂度**：每个头的注意力计算都需要独立进行，因此使用较高维度的注意力机制可能会导致计算量大幅增加。通过将每个头的注意力输出进行降维，可以有效降低计算复杂度，使模型更容易进行训练和部署。
   > 2. **提高模型的表达能力**：每个头所关注的信息可能不同，通过对每个头进行降维，可以使每个头能够更好地捕捉到不同方面的信息。这样可以增加模型的表达能力，使其能够更好地适应不同的输入数据和任务。

7. 大概讲一下Transformer的Encoder模块？

   > 参考 [Transformer中Encoder-Decoder实现中的Encoder](#Encoder)

8. 为何在获取输入词向量之后需要对矩阵乘以embedding size的开方？意义是什么？

   > 参考 除以$\sqrt{d_k}$  的回答。在获取输入词向量之后，对矩阵乘以词向量维度的开方的目的是为了对词向量进行缩放，以使得输入数据的范围在一个合适的区间内，进而帮助模型更好地学习。

9. 简单介绍一下Transformer的位置编码？有什么意义和优缺点？

   > 参考 [嵌入层Position Encoding](# 嵌入层Position Encoding)

10.  你还了解哪些关于位置编码的技术，各自的优缺点是什么？

    > 1. **Sinusoidal Positional Encoding（正弦位置编码）**：
    >    - 优点：
    >      - 简单易实现。
    >      - 能够编码序列中每个位置的绝对位置信息。
    >    - 缺点：
    >      - 只能捕捉到位置信息，缺少语义信息。
    >      - 对于较长的序列可能不够精确。
    > 2. **Learned Positional Embeddings（学习位置嵌入）**：
    >    - 优点：
    >      - 能够学习到更复杂的位置信息和语义信息。
    >      - 可以根据任务和数据自动调整位置编码。
    >    - 缺点：
    >      - 需要额外的参数学习。
    >      - 可能需要更多的训练数据和计算资源。
    > 3. **Transformer-XL Positional Encodings（Transformer-XL 位置编码）**：
    >    - 优点：
    >      - 能够在长序列中更好地捕捉到位置信息。
    >      - 通过重复使用固定的位置编码，节省了计算成本。
    >    - 缺点：
    >      - 在处理非常长的序列时，可能会出现信息衰减的问题。

11. 为什么transformer块使用LayerNorm而不是BatchNorm？LayerNorm 在Transformer的位置是哪里？

    > 1. **独立性**：LayerNorm 是对每个样本的特征进行归一化，而 BatchNorm 则是对每个特征维度进行归一化。在 Transformer 模型中，每个位置的特征都可以被视为一个样本，因此 LayerNorm 更适合用于这种场景，能够保持样本之间的独立性。
    > 2. **稳定性**：LayerNorm 对于小批量大小更加稳定，因为它不依赖于批量内的统计信息，而是使用每个样本的特征值来计算均值和方差。
    > 3. **参数共享**：LayerNorm 有助于参数共享，因为它对每个样本的特征都采用相同的均值和方差计算方式，这样有助于模型的训练和泛化。

12. 简答讲一下BatchNorm技术，以及它的优缺点。

    > Batch Normalization (BatchNorm) 主要思想是通过对每个批次的输入数据进行归一化处理，以减少网络训练过程中的内部协变量偏移，进而加速收敛和提高泛化性能。
    >
    > **优点**：
    >
    > 1. **加速收敛**：BatchNorm 有助于加速神经网络的训练，因为它减少了网络在训练过程中的内部协变量偏移，使得网络更容易收敛。
    > 2. **减少梯度消失/爆炸**：BatchNorm 能够减少梯度消失或爆炸的问题，使得网络更容易训练和优化。
    > 3. **降低对初始参数的敏感性**：BatchNorm 能够减少网络对初始参数的敏感性，使得网络更容易初始化，并且不太依赖于初始参数的选择。
    > 4. **正则化效果**：BatchNorm 在一定程度上起到了正则化的作用，有助于防止过拟合。
    >
    > **缺点**：
    >
    > 1. **增加计算成本**：BatchNorm 需要对每个批次的输入数据进行归一化处理，这会增加一定的计算成本。
    > 2. **不适用于小批量大小**：在小批量大小下，BatchNorm 的效果可能会变差，甚至会出现数值不稳定的问题。
    > 3. **对 RNN 和 CNN 不够适用**：BatchNorm 在处理循环神经网络（RNN）和卷积神经网络（CNN）时效果可能不如全连接神经网络明显。

13. Decoder阶段的多头自注意力和encoder的多头自注意力有什么区别？

    > 1. **查询和键的来源**：
    >    - 在 Encoder 阶段的多头自注意力中，查询（Q）和键（K）都来自于 Encoder 的输出。
    >    - 而在 Decoder 阶段的多头自注意力中，查询（Q）也是来自 Decoder 自身的输出，而键（K）则仍来自于 Encoder 的输出。
    > 2. **位置编码**：
    >    - 在 Encoder 阶段的多头自注意力中，通常会将位置编码添加到 Encoder 的输入序列中，以提供关于输入序列中单词位置的信息。
    >    - 在 Decoder 阶段的多头自注意力中，除了添加位置编码到 Decoder 的输入序列中外，还需要在注意力计算中使用序列掩码（sequence mask）来确保模型只关注当前位置之前的信息，以避免看到未来的信息。

14. Transformer的并行化提现在哪个地方？Decoder端可以做并行化吗？

    > Transformer 模型的并行化主要体现在两个方面：
    >
    > 1. **自注意力机制的并行计算**：
    >    - 在 Encoder 和 Decoder 部分的自注意力机制中，可以将每个头的注意力权重矩阵并行计算，然后合并得到最终的注意力矩阵。这种并行计算方式能够大大提高模型的计算效率。
    > 2. **批量化的并行计算**：
    >    - 在处理输入数据时，Transformer 模型通常会将输入数据分成多个批次，每个批次中包含多个样本。在每个批次中，模型会同时处理多个样本，从而实现了数据的批量化并行计算。
    >
    > 对于 Decoder 端，虽然可以进行一定程度的并行化，但相比于 Encoder 端来说，其并行化程度较低。主要的原因包括：
    >
    > 1. **序列依赖性**：
    >    - 在解码过程中，每个输出位置的计算通常依赖于前面位置的输出结果。因此，虽然可以在一定程度上并行计算不同位置的输出，但仍然存在着一定的依赖关系，限制了并行化的程度。
    > 2. **自注意力的计算顺序**：
    >    - 在 Decoder 端的自注意力计算中，需要按顺序处理每个位置的输出，以确保每个位置都能正确地获取其前面位置的信息。这种计算顺序限制了 Decoder 端自注意力计算的并行化程度。
    >
    > 虽然在某些情况下可以对 Decoder 进行一定程度的并行化，例如通过将多个解码器堆叠在一起并行处理不同样本的解码过程，或者使用束搜索（beam search）等技术来并行处理多个候选输出序列，但相比于 Encoder 端来说，并行化程度较低。

15. 简单描述一下wordpiece model 和 byte pair encoding，有实际应用过吗？

    > WordPiece 模型和 Byte Pair Encoding (BPE) 都是常用的无监督分词技术，用于将文本数据分割成词语或子词单元。
    >
    > 1. **WordPiece 模型**：
    >    - WordPiece 模型是由 Google 提出的一种基于子词级别的分词方法。它将词汇表中的每个词都视为一个初始的单词（或子词单元），然后通过迭代地将最频繁出现的词或子词进行拆分，直到达到指定的词表大小或者停止条件。
    >    - WordPiece 模型基于统计信息，可以根据语料库中的频率来动态地调整词汇表，以适应不同的任务和语言。
    > 2. **Byte Pair Encoding (BPE)**：
    >    - Byte Pair Encoding 是由 Sennrich 等人提出的一种基于字符级别的分词方法。它将文本数据中的字符作为初始的单词（或子词单元），然后通过不断地合并频率最高的相邻字符对，直到达到指定的词表大小或者停止条件。
    >    - BPE 算法通过动态地合并字符对来构建词汇表，能够适应不同的语言和文本数据，并且不需要使用外部语料库。
    >
    > 这两种分词方法都被广泛应用于自然语言处理任务中，例如机器翻译、文本生成、语音识别等。它们能够将文本数据有效地分割成更小的单元，有助于提高模型的泛化能力和性能。
    >
    > 个人经验上，我在一些自然语言处理项目中使用过 BPE 进行分词和子词划分，特别是在处理非常规或者低资源语言时，BPE 能够提供更好的效果。

16. 引申一个关于bert问题，bert的mask为何不学习transformer在attention处进行屏蔽score的技巧？

    > BERT中的masking策略与Transformer中的attention屏蔽(score masking)技巧有所不同，主要有以下几个原因：
    >
    > 1. **训练和推断的一致性**：在Transformer的self-attention中，为了提高模型的泛化能力，训练时需要随机屏蔽一些token，而在推断时不需要进行屏蔽。然而，在BERT中，由于是基于双向语境的，所以训练和推断时都需要使用mask，以确保模型能够正确地预测缺失的token。
    > 2. **目标不同**：在Transformer中的attention屏蔽(score masking)技巧主要是为了使模型在训练时不能看到后续位置的信息，以更好地学习到序列信息的顺序。而在BERT中，masking的目的是为了使模型学习到对缺失token的预测能力，即对于被mask的token进行正确的预测。
    > 3. **预训练和微调的差异**：BERT模型是通过预训练和微调两个阶段来完成任务的。在预训练阶段，masking是随机应用的，以促使模型学习到上下文信息。在微调阶段，模型会接收到完整的输入，并根据任务的要求进行微调。
    >
    > 综上所述，BERT中的masking策略与Transformer中的attention屏蔽(score masking)技巧有所不同，==主要是由于它们的训练目标和场景不同。BERT的masking主要是为了让模型学会预测缺失的token，从而提高模型的泛化能力==。



#### Bert

> 详见：[Bert](#Bert)
