import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, heads, d, k, m, dropout=0.):
        super(TransformerBlock, self).__init__()
        self.k = k
        self.heads = heads

        self.wq = nn.Linear(d, heads*k, bias=False)
        self.wk = nn.Linear(d, heads*k, bias=False)
        self.wv = nn.Linear(d, heads*k, bias=False)
        self.wc = nn.Linear(heads*k, d, bias=False)
        self.dropoutatt = nn.Dropout(dropout)

        self.w1 = nn.Linear(d, m)
        self.dropoutfc = nn.Dropout(dropout)
        self.w2 = nn.Linear(m, d)

        self.layernorm1 = nn.LayerNorm(d)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(d)
        self.dropout2 = nn.Dropout(dropout)

        nn.init.normal_(self.wq.weight, 0, .02)
        nn.init.normal_(self.wk.weight, 0, .02)
        nn.init.normal_(self.wv.weight, 0, .02)
        nn.init.normal_(self.wc.weight, 0, .02)

        nn.init.normal_(self.w1.weight, 0, .02)
        nn.init.constant_(self.w1.bias, 0.0)
        nn.init.normal_(self.w2.weight, 0, .02)
        nn.init.constant_(self.w2.bias, 0.0)

    def forward(self, x, mask):

        seq_len, batch_size, embed_dim = x.shape

        # --- Multi-Head Self-Attention ---

        # 1. Project x into Q, K, V
        Q = self.wq(x)  # (seq_len, batch_size, heads*k)
        K = self.wk(x)  # (seq_len, batch_size, heads*k)
        V = self.wv(x)  # (seq_len, batch_size, heads*k)

        # 2. Reshape to split heads
        #    New shape: (seq_len, batch_size, heads, k) -> permute -> (batch_size*heads, seq_len, k)
        Q = Q.view(seq_len, batch_size, self.heads, self.k).permute(1, 2, 0, 3).reshape(batch_size*self.heads, seq_len, self.k)
        K = K.view(seq_len, batch_size, self.heads, self.k).permute(1, 2, 0, 3).reshape(batch_size*self.heads, seq_len, self.k)
        V = V.view(seq_len, batch_size, self.heads, self.k).permute(1, 2, 0, 3).reshape(batch_size*self.heads, seq_len, self.k)

        # 3. Scaled dot-product attention: (batch_size*heads, seq_len, seq_len)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.k)

        # 4. Apply the mask (broadcasts across the first dimension)
        #    mask is (seq_len, seq_len), attn_scores is (batch_size*heads, seq_len, seq_len)
        attn_scores += mask

        # 5. Softmax over the last dimension, then dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropoutatt(attn_weights)

        # 6. Multiply by V to get the output
        attn_out = torch.bmm(attn_weights, V)  # (batch_size*heads, seq_len, k)

        # 7. Reshape back to (seq_len, batch_size, heads*k)
        attn_out = attn_out.view(batch_size, self.heads, seq_len, self.k)
        attn_out = attn_out.permute(2, 0, 1, 3).reshape(seq_len, batch_size, self.heads * self.k)

        # 8. Final linear projection back to d dimensions
        attn_out = self.wc(attn_out)

        # 9. Skip connection + LayerNorm
        x = x + self.dropout1(attn_out)
        x = self.layernorm1(x)

        # --- Position-wise Feed-Forward ---

        ff = self.w1(x)         # (seq_len, batch_size, m)
        ff = F.relu(ff)         # non-linearity
        ff = self.dropoutfc(ff) # dropout
        ff = self.w2(ff)        # (seq_len, batch_size, d)
        ff = self.dropout2(ff)

        #  Skip connection + LayerNorm
        out = x + ff
        out = self.layernorm2(out)

        return out


class Transformer(nn.Module):
    def __init__(self, seq_len, tokens, d, k, m, heads, layers, tied_weights=False, dropout=0., dropoutio=0.):
        super(Transformer, self).__init__()
        self.mask = None
        self.pos = None
        self.dims = d
        self.tied_weights = tied_weights
        self.dropout=dropout

        self.positional_embedding = nn.Embedding(seq_len, d)
        self.dropi = nn.Dropout(dropoutio)
        self.word_embedding = nn.Embedding(tokens, d)
        self.transformer = nn.ModuleList()
        for i in range(layers):
            self.transformer.append(TransformerBlock(heads, d, k, m, dropout))

        if not tied_weights: self.decoder = nn.Linear(d, tokens)
        self.dropo = nn.Dropout(dropoutio)
        self.bias = nn.Parameter(torch.ones(tokens))

        nn.init.normal_(self.positional_embedding.weight, 0, .02)
        nn.init.normal_(self.word_embedding.weight, 0, .02)
        if not self.tied_weights: nn.init.normal_(self.decoder.weight, 0, .02)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x):
        if self.mask is None or self.mask.shape[0] != x.shape[0]:
            self.mask = torch.triu(torch.ones(len(x), len(x)))
            self.mask.masked_fill_(self.mask == 0, float('-inf')).masked_fill_(self.mask == 1, float(0.0))
            self.mask = self.mask.transpose(0,1).to(x.device)
            self.pos = torch.arange(0, x.shape[0], dtype=torch.long).to(x.device)

        x = self.word_embedding(x) * math.sqrt(self.dims)
        p = self.positional_embedding(self.pos)[:,None,:]
        z = F.relu(self.dropi(x) + self.dropi(p))
        for layer in self.transformer:
            z = layer(z, self.mask)

        z = self.dropo(z)
        outputs = torch.matmul(z, self.word_embedding.weight.t()) if self.tied_weights else self.decoder(z)
        return F.log_softmax(outputs + self.bias, dim=-1)

