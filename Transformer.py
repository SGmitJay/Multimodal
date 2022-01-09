import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math


## SG
def make_batch(sentences):
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]

    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)

# get_attn_pad_mask
# 这里需要有有一个同样大小的矩阵，告诉我们哪个位置是PAD部分，之后在计算softmax之前会把这里设置为无穷大
# 这里得到的矩阵形状是batch_size x len_q x len_k
# seq_q和seq_k不一定一致，在交互注意力，q来自解码端，k来自编码端，所以告诉模型编码这边的pad符号信息就可以，解码端的pad信息在交互注意力层是没有用到的
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)   # batch_size x 1 x len_k,one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)    # batch_size x len_q x len_k

def get_attn_subsequent_mask(seq):
    """

    seq:[batch_size,tgt_len]

    """
    # attn_shape:[batch_size,tgt_len,tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]

    subsequence_mask = np.triu(np.ones(attn_shape), k=1)    # 生成一个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask   # [batch_size,tgt_len,tgt_len]



# Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 这里定义了一个词向量矩阵[src_vocab_size,d_model]
        self.src_emb = nn.Embedding(src_vocab_size, d_model)      # 词嵌入层
        self.pos_emb = PositionalEncoding(d_model)      # 位置编码，这里是采用固定的正余弦函数，也可以使用类似词向量的nn.Embedding获得一个可以更新学习的位置编码
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        # 使用ModuleList对多个encoder进行堆叠，因为后续的encoder并没有使用词向量和位置编码，所以抽离出来
        # 这里将自注意力和前馈神经网络合并在一起
    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)

        # get_attn_pad_mask是为了得到句子中的pad的位置信息，给到模型后面，在计算自注意力和交互注意力的时候去掉pad符号的影响
        enc_self_attns_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attns_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
    def forward(self, dec_inputs, enc_inputs, enc_outputs):   # dec_inputs:[batch_size,target_len]
        dec_outputs = self.tgt_emb(dec_inputs)    # dec_outputs:[batch_size,tgt_len,d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)

        # get_attn_pad_mask 自注意力层的pad部分
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        # get_attn_subsequent_mask 这个做的是自注意层的mask部分，就是当前单词之后看不到，使用一个上三角为1的矩阵
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        # 两个矩阵相加，大于0的为1，不大于0的为0，为1的在之后就会被fill到无限小
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask+dec_self_attn_subsequent_mask), 0)

        # 这个是交互注意力机制中的mask矩阵，enc的输入是k,主要关注k中哪些是pad符号，给到后面的模型
        # q也是有pad符号，但是这里不关注 ？？？
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

# DecoderLayer包含两个部分
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn



# EncoderLayer包含两个部分，多个注意力机制和前馈神经网络
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer,self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
    def forward(self, enc_inputs, enc_self_attn_mask):
        # 下面这个就是自注意力层，输入是enc_inputs,形状是[batch_size x seq_len_q x d_model] 需要注意的是最初始的QKV矩阵是等同于这个输入的
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

# PoswiseFeedForwardNet()
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs    # inputs:[batch_size,len_q,d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output+residual)





# MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention,self).__init__()
        # 输入进来的Q，K，V是相等，我们会使用映射linear做一个映射得到参数矩阵W_Q,W_K,W_V
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        # 这个多头分为这几个步骤，首先映射分头，然后计算atten_scores，然后计算atten_value
        # Q:[batch_size x len_q x d_model],K:[batch_size x len_k x d_model],V:[batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # 下面就是先映射，后分头，一定要注意q和k分头之后维度是一致的
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # v_s: [batch_size x n_heads x len_k x d_v]

        # 输入进来的attn_mask形状[batch_size,len_q,len_k]
        # 经过下面的代码得到新的attn_mask :[batch_size,n_heads,len_q,len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # 计算ScaledDotProductAttention这个函数

        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        # context: [batch_size,len_q,n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads*d_v)
        output = self.linear(context)   # output:[batch_size,len_q,d_model]
        return self.layer_norm(output+residual), attn

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # Q:[batch_size,n_heads,len_q,d_k] K:[batch_size,n_heads,len_k,d_k] V:[batch_size,n_heads,len_k,d_v]
        # scores:[batch_size,n_heads,len_q,len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2))/np.sqrt(d_k)

        # 下面就使用了attn_mask，把被mask的地方置为无限小，softmax之后就是0，对q的单词不起作用
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn







# PositionalEncoding 代码实现
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # pos代表的是单词在句子的索引。
        # 假设d_model是512，2i那个符号中i从0取到了255，那么2i对应的值就是0，2，4....510
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)

        # 上面的代码获取之后得到的pe的形状是:[max_len,d_model]

        # 下面这个代码之后，得到的pe形状是[max_len,1,d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)

        # 定一个缓冲区，其简单理解为这个参数不更新就可以
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x:[seq_len,batch_size,d_model]
        """
        x = x+self.pe[:x.size(0), :]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()   # 编码层
        self.decoder = Decoder()   # 解码层
        # d_modal是解码层每个token输出的维度大小，之后会做一个tgt_vocab_size大小的softmax
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
    def forward(self , enc_inputs, dec_inputs):
        # 两个输入
        # 编码端输入enc_inputs,[batch_size,src_len]
        # 解码端输入dec_inputs,[batch_size,tgt_len]

        # enc_outputs是主要的输出，enc_self_attns是QK转置相乘之后softmax之后的矩阵值，代表的是每个单词和其他单词的相关性
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns=self.decoder(dec_inputs, enc_inputs, enc_outputs)

        # dec_outputs映射到词表大小
        dec_logits=self.projection(dec_outputs)   # dec_logits:[batch_size x src_vocab_size x tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


if __name__ == '__main__':
    # 句子的输入部分
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
    # 构建词表
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    src_vocab_size=len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5
    tgt_len = 5
    # 模型参数
    d_model = 512   # Embedding Size
    d_ff = 2048    # FeedForward dimension
    d_k = d_v = 64   # dimension of K(=Q),V
    n_layers = 6   # number of Encoder and Decoder layer
    n_heads = 8     # number of heads in Multi-Head Attention

    model = Transformer()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    enc_inputs, dec_inputs, target_batch = make_batch(sentences)

    for epoch in range(20):
        optimizer.zero_grad()
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)

        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('Epoch:', '%04d' % (epoch+1), 'cost=', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()





