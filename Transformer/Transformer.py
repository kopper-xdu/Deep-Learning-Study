import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # 希望模型中的某些参数不更新，又希望参数保存下来
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, len_q, len_k]
        说明:在计算Cross Attention层中len_q(q1,..qt)和len_k(k1,...km)可能不同
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Q.size(-1)) # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -torch.inf)   # Fills elements of self tensor with value where mask is True.TODO

        attn = F.softmax(scores, dim=-1)

        context = torch.matmul(attn, V)   # context: [batch_size, n_heads, len_q, d_v]

        return context, attn   # attn注意力矩阵（用于可视化）

class MultiHeadAttention(nn.Module):
    """
    这个Attention类可以实现:
    Encoder的Self-Attention
    Decoder的Masked Self-Attention
    Encoder-Decoder的Attention
    """
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head

        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """

        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(input_Q.size(0), input_Q.size(1), self.n_head, -1).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k]
        K = self.W_K(input_K).view(input_K.size(0), input_K.size(1), self.n_head, -1).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(input_V.size(0), input_V.size(1), self.n_head, -1).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)  #TODO
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)

        context = context.transpose(1, 2).contiguous().view(context.size(0), Q.size(2), -1)

        output = self.fc(context)

        return self.layer_norm(output + input_Q), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.fc = nn.Sequential(nn.Linear(d_model, d_ff, bias=False),
                                nn.ReLU(),
                                nn.Linear(d_ff, d_model, bias=False))
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        output = self.fc(x)
        return self.layer_norm(output + x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_head)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]  mask矩阵(pad mask or sequence mask)
        """
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff):
        super().__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, n_head)
        self.dec_enc_attn = MultiHeadAttention(d_model, n_head)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs,
                                                        dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs,
                                                      dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn


def get_attn_pad_mask(seq_q, seq_k):
    # pad mask的作用：在对value向量加权平均的时候，可以让pad对应的alpha_ij=0，这样注意力就不会考虑到pad向量
    """
    seq_q: [batch_size, len_q]
    seq_k: [batch_size, len_k]
    """
    batch_size, len_q = seq_q.size()  # 这个seq_q只是用来expand维度的
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # 例如:seq_k = [[1,2,3,4,0], [1,2,3,5,0]]
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k).to(device)  # [batch_size, len_q, len_k]

def get_attn_subsequence_mask(seq):
    """
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = torch.triu(torch.ones(attn_shape)).eq(0)  # 生成一个上三角矩阵
    return subsequence_mask.to(device)  # [batch_size, tgt_len, tgt_len]


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, n_head, d_ff, n_layers):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)  # token Embedding
        self.pos_emb = PositionalEncoding(d_model)  # Transformer中位置编码时固定的，不需要学习
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_head, d_ff) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len]
        """
        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        # Encoder输入序列的pad mask矩阵
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        enc_self_attns = []  # 在计算中不需要用到，它主要用来保存你接下来返回的attention的值（这个主要是为了你画热力图等，用来看各个词之间的关系
        for layer in self.layers:  
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs,
                                               enc_self_attn_mask)  # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention
            enc_self_attns.append(enc_self_attn)  # 这个只是为了可视化
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, n_head, d_ff, n_layers):
        super().__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_head, d_ff) for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        dec_inputs: [batch_size, tgt_len]
        enc_inputs: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]
        """
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, tgt_len, d_model]
        # Decoder输入序列的pad mask矩阵（这个例子中decoder是没有加pad的，实际应用中都是有pad填充的）
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)  # [batch_size, tgt_len, tgt_len]
        # Masked Self_Attention：当前时刻是看不到未来的信息的
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)  # [batch_size, tgt_len, tgt_len]

        # Decoder中把两种mask矩阵相加（既屏蔽了pad的信息，也屏蔽了未来时刻的信息）
        dec_self_attn_mask = torch.gt(dec_self_attn_pad_mask + dec_self_attn_subsequence_mask,
                                      0)  # [batch_size, tgt_len, tgt_len]; torch.gt比较两个矩阵的元素，大于则返回1，否则返回0

        # 这个mask主要用于encoder-decoder attention层
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)  # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            # Decoder的Block是上一个Block的输出dec_outputs（变化）和Encoder网络的输出enc_outputs（固定）
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_head, d_ff, n_layers):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_head, d_ff, n_layers)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_head, d_ff, n_layers)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        """
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        """
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)

        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns