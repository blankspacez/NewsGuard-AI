import os
import pickle
import re
from typing import Dict, Any, Optional

import jieba
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class TextProcessor:
    """
    文本处理与张量化
    """

    def __init__(self, lang: str, vocab_path: str, max_len: int = 50, stopwords_path: Optional[str] = None):
        self.lang = lang
        self.max_len = max_len
        self.vocabulary = self._load_vocab(vocab_path)
        self.stopwords = self._load_stopwords(stopwords_path)
        print(f"[{lang}] Vocab size: {len(self.vocabulary)}")

    def _load_vocab(self, path: str) -> Dict[str, int]:
        if not os.path.exists(path):
            print(f"Error: Vocab file not found at {path}")
            return {'<UNK>': 1}
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _load_stopwords(self, path: Optional[str]) -> set[str]:
        s = set()
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    s.add(line.strip())
        return s

    def clean_and_tokenize(self, text: str) -> list[str]:
        text = text.lower().strip()
        tokens: list[str] = []
        if self.lang == 'zh':
            text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text)
            tokens = jieba.lcut(text)
        else:
            text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
            text = re.sub(r"\'s", " \'s", text)
            text = re.sub(r"\s{2,}", " ", text)
            tokens = text.split()

        if self.stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        return tokens

    def transform(self, text: str) -> torch.Tensor:
        tokens = self.clean_and_tokenize(text)
        seq = [self.vocabulary.get(t, 0) for t in tokens]
        if len(seq) >= self.max_len:
            seq = seq[:self.max_len]
        else:
            seq = [0] * (self.max_len - len(seq)) + seq
        return torch.tensor([seq], dtype=torch.long)


class TransformerBlock(nn.Module):
    def __init__(self, input_size, d_k=16, d_v=16, n_heads=8, is_layer_norm=False, attn_dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size
        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)
        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * d_v))
        self.W_o = nn.Parameter(torch.Tensor(d_v * n_heads, input_size))
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()
        # 存储最近一次前向得到的多头 self-attention，用于可解释性
        self.last_attn: Optional[torch.Tensor] = None  # (bsz, q_len, k_len)

    def __init_weights__(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        output = self.linear2(F.relu(self.linear1(X)))
        output = self.dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, bsz, q_len, k_len, episilon=1e-6):
        temperature = self.d_k ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)
        Q_K_score = F.softmax(Q_K, dim=-1)

        # 还原回 (bsz, n_heads, q_len, k_len)，再对多头取平均，得到 (bsz, q_len, k_len)
        n_heads = self.n_heads
        attn_heads = Q_K_score.view(bsz, n_heads, q_len, k_len)
        # 在多头维度上做平均，得到单头等效注意力矩阵
        self.last_attn = attn_heads.mean(dim=1).detach()

        Q_K_score = self.dropout(Q_K_score)
        return Q_K_score.bmm(V)

    def multi_head_attention(self, Q, K, V):
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()
        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)
        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz * self.n_heads, q_len, self.d_v)
        V_att = self.scaled_dot_product_attention(Q_, K_, V_, bsz, q_len, k_len)
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v).permute(0, 2, 1, 3).contiguous().view(
            bsz, q_len, self.n_heads * self.d_v
        )
        return self.dropout(V_att.matmul(self.W_o))

    def forward(self, Q, K, V):
        V_att = self.multi_head_attention(Q, K, V)
        if self.is_layer_norm:
            X = self.layer_morm(Q + V_att)
            return self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            return self.FFN(X) + X


class ResNet50_Encoder(nn.Module):
    def __init__(self):
        super(ResNet50_Encoder, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(2048, 300)

    def forward(self, x):
        return self.model(x)


class StudentNN(nn.Module):
    def __init__(self, config: Dict[str, Any], vocab_size: int):
        super(StudentNN, self).__init__()
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        embedding_weights = config['embedding_weights']
        V, D = embedding_weights.shape
        maxlen = config['maxlen']
        dropout_rate = config['dropout']

        self.mh_attention = TransformerBlock(input_size=300, n_heads=8, attn_dropout=0)
        self.word_embedding = nn.Embedding(
            num_embeddings=V, embedding_dim=D, padding_idx=0, _weight=torch.from_numpy(embedding_weights)
        )
        self.image_embedding = ResNet50_Encoder()
        self.convs = nn.ModuleList([nn.Conv1d(300, 100, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen - K + 1) for K in config['kernel_sizes']])
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(600, 300)
        self.fc2 = nn.Linear(in_features=300, out_features=config['num_classes'])
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)

    def _forward_internal(self, X_text, X_image, return_features: bool = False):
        X_text = self.word_embedding(X_text)
        if self.config['user_self_attention'] is True:
            X_text = self.mh_attention(X_text, X_text, X_text)
        X_text = X_text.permute(0, 2, 1)

        iembedding = self.image_embedding(X_image)

        conv_block = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(X_text))
            pool = max_pooling(act)
            pool = pool.squeeze(-1)
            conv_block.append(pool)
        text_feature = torch.cat(conv_block, dim=1)
        bsz = text_feature.size()[0]

        # text_feature: [B, 300], iembedding: [B, 300]

        self_att_t = self.mh_attention(
            text_feature.view(bsz, -1, 300),
            text_feature.view(bsz, -1, 300),
            text_feature.view(bsz, -1, 300),
        )
        self_att_i = self.mh_attention(
            iembedding.view(bsz, -1, 300), iembedding.view(bsz, -1, 300), iembedding.view(bsz, -1, 300)
        )

        co_att_ti = self.mh_attention(self_att_t, self_att_i, self_att_i).view(bsz, 300)
        co_att_it = self.mh_attention(self_att_i, self_att_t, self_att_t).view(bsz, 300)

        att_feature = torch.cat((co_att_ti, co_att_it), dim=1)
        a1 = self.relu(self.dropout(self.fc1(att_feature)))
        d1 = self.dropout(a1)
        output = self.fc2(d1)

        if return_features:
            # 返回融合前的 300 维文本/图像特征
            return output, text_feature, iembedding
        return output

    def forward(self, X_text, X_image):
        # 默认推理接口，保持向后兼容
        return self._forward_internal(X_text, X_image, return_features=False)

    def get_text_self_attention(self, X_text: torch.Tensor) -> Optional[torch.Tensor]:
        """
        提取文本 self-attention 矩阵:
        - 输入: X_text [bsz, L]（与 forward 中相同的索引张量）
        - 输出: attn [bsz, L, L]，在多头维度上已做平均
        """
        self.eval()
        with torch.no_grad():
            X_embed = self.word_embedding(X_text)          # [bsz, L, 300]
            # 这里只做一次自注意力，以便 TransformerBlock 记录 last_attn
            _ = self.mh_attention(X_embed, X_embed, X_embed)
            return self.mh_attention.last_attn  # (bsz, L, L)

