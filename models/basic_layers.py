import torch
from torch import nn, einsum
from einops import rearrange, repeat
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    def __init__(self, modality1_dim, modality2_dim, embed_dim, attn_dropout=0.5):
        super(CrossModalAttention, self).__init__()
        self.modality1_dim = modality1_dim
        self.modality2_dim = modality2_dim
        self.embed_dim = embed_dim
        self.modality1_ln = nn.LayerNorm(self.modality1_dim)
        self.modality2_ln = nn.LayerNorm(self.modality2_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.scaling = self.embed_dim ** -0.5
        self.proj_modality1 = nn.Linear(self.modality1_dim, self.embed_dim)
        self.proj_modality2_k = nn.Linear(self.modality2_dim, self.embed_dim)
        self.proj_modality2_v = nn.Linear(self.modality2_dim, self.embed_dim)
        self.proj = nn.Linear(self.embed_dim, self.modality1_dim)

    def forward(self, modality1, modality2):
        q = self.proj_modality1(self.modality1_ln(modality1))
        k = self.proj_modality2_k(self.modality2_ln(modality2))
        v = self.proj_modality2_v(self.modality2_ln(modality2))
        attention = F.softmax(torch.bmm(q, k.permute(0, 2, 1)) * self.scaling, dim=-1)
        context = torch.bmm(attention, v)
        output = self.proj(context)
        return output

class CrossmodalEncoderLayer(nn.Module):
    def __init__(self, proxy_dim, text_dim, audio_dim, video_dim, embed_dim, attn_dropout=0.5, args=None):
        super(CrossmodalEncoderLayer, self).__init__()
        self.cma_t = CrossModalAttention(proxy_dim, text_dim, embed_dim)
        self.cma_a = CrossModalAttention(proxy_dim, audio_dim, embed_dim)
        self.cma_v = CrossModalAttention(proxy_dim, video_dim, embed_dim)

        self.layernorm = nn.LayerNorm(proxy_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.fc = nn.Linear(proxy_dim, proxy_dim)
    def forward(self, proxy_m, text, audio, video,weight_t_v_a):
        if proxy_m is None:
            output = self.cma_a(text, audio) + self.cma_v(text, video) + text
        else:
            output = (self.cma_t(proxy_m, text) * weight_t_v_a[0] +
                      self.cma_a(proxy_m, audio) * weight_t_v_a[2] +
                      self.cma_v(proxy_m, video) * weight_t_v_a[1] +proxy_m)
        residual = output
        output = self.fc(self.layernorm(output))
        output = self.attn_dropout(output)
        output = output + residual
        return output



class CrossmodalEncoder(nn.Module):
    def __init__(self, proxy_dim,text_dim, audio_dim, video_dim, embed_dim, num_layers=4, attn_dropout=0.5):
        super(CrossmodalEncoder, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([])
        for layer in range(self.num_layers):
            self.layers.append(CrossmodalEncoderLayer(proxy_dim,text_dim, audio_dim, video_dim, embed_dim, attn_dropout))

    def forward(self, proxy_m, text, audio, video,weight_t_v_a):
        if proxy_m is None:
            output = text
            for layer in self.layers:
                output = layer(None, text, audio, video,weight_t_v_a)
            return output
        output = proxy_m
        for layer in self.layers:
            output = layer(output, text, audio, video,weight_t_v_a)
        return output




class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm_qkv(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)

        return self.fn(q, k, v)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads

        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_qkv(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x, save_hidden=False):
        if save_hidden == True:
            hidden_list = []
            hidden_list.append(x)
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
                hidden_list.append(x)
            return hidden_list
        else:
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
            return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_qkv(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm_qkv(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, tgt, memory):
        for attn1, attn2, ff in self.layers:
            tgt = attn1(tgt, tgt, tgt) + tgt
            tgt = attn1(tgt, memory, memory) + tgt
            tgt = ff(tgt) + tgt
        return tgt



class CrossTransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_qkv(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, source_x, target_x):
        for attn, ff in self.layers:
            target_x_tmp = attn(target_x, source_x, source_x)
            target_x = target_x_tmp + target_x
            target_x = ff(target_x) + target_x
        return target_x



class Transformer(nn.Module):
    def __init__(self, *, num_frames, token_len, save_hidden, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.token_len = token_len
        self.save_hidden = save_hidden

        if token_len is not None:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames + token_len, dim))
            self.extra_token = nn.Parameter(torch.zeros(1, token_len, dim))
        else:
             self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
             self.extra_token = None

        self.dropout = nn.Dropout(emb_dropout)

        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()


    def forward(self, x):
        b, n, _ = x.shape

        if self.token_len is not None:
            extra_token = repeat(self.extra_token, '1 n d -> b n d', b = b)
            x = torch.cat((extra_token, x), dim=1)
            x = x + self.pos_embedding[:, :n+self.token_len]
        else:
            x = x + self.pos_embedding[:, :n]

        x = self.dropout(x)
        x = self.encoder(x, self.save_hidden)

        return x



