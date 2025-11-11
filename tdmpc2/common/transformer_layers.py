"""Transformer layers for In-Context Learning in TD-MPC2

Implements:
- CausalSelfAttention: 未来を見ないself-attention
- TransformerBlock: Transformer decoder block
- PositionalEncoding: Sinusoidal positional encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt, log


class CausalSelfAttention(nn.Module):
	"""Causal（未来を見ない）Self-Attention with K-V caching
	
	Args:
		embed_dim: 埋め込み次元
		n_heads: アテンションヘッド数
		dropout: ドロップアウト率
		max_seq_len: 最大シーケンス長（causal mask用）
	"""
	def __init__(self, embed_dim, n_heads, dropout=0.1, max_seq_len=1024):
		super().__init__()
		assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
		
		self.embed_dim = embed_dim
		self.n_heads = n_heads
		self.head_dim = embed_dim // n_heads
		self.scale = 1.0 / sqrt(self.head_dim)
		
		# Q, K, V projection (一括で計算)
		self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
		self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
		self.attn_dropout = nn.Dropout(dropout)
		self.proj_dropout = nn.Dropout(dropout)
		
		# Causal mask (下三角行列)
		self.register_buffer(
			"causal_mask",
			torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
		)
		
	def forward(self, x, use_cache=False, past_kv=None):
		"""
		Args:
			x: (batch, seq_len, embed_dim)
			use_cache: K-Vキャッシュを使用するか（プランニング高速化用）
			past_kv: 過去のK, V - tuple of (K, V)
				K: (batch, n_heads, past_len, head_dim)
				V: (batch, n_heads, past_len, head_dim)
			
		Returns:
			out: (batch, seq_len, embed_dim)
			cache: (K, V) for next step (if use_cache=True)
		"""
		B, T, C = x.shape
		
		# Q, K, V を計算
		qkv = self.qkv(x)  # (B, T, 3*C)
		qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
		qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, head_dim)
		q, k, v = qkv[0], qkv[1], qkv[2]
		
		# K-Vキャッシュ適用（プランニング時の高速化）
		if use_cache and past_kv is not None:
			past_k, past_v = past_kv
			k = torch.cat([past_k, k], dim=2)  # (B, n_heads, past_len+T, head_dim)
			v = torch.cat([past_v, v], dim=2)
		
		# Scaled dot-product attention
		kv_len = k.size(2)
		att = (q @ k.transpose(-2, -1)) * self.scale  # (B, n_heads, T, kv_len)
		
		# Causal mask適用
		att = att.masked_fill(self.causal_mask[:, :, :T, :kv_len] == 0, float('-inf'))
		
		att = F.softmax(att, dim=-1)
		att = self.attn_dropout(att)
		
		out = att @ v  # (B, n_heads, T, head_dim)
		out = out.transpose(1, 2).contiguous().view(B, T, C)
		out = self.proj(out)
		out = self.proj_dropout(out)
		
		cache = (k, v) if use_cache else None
		return out, cache


class TransformerBlock(nn.Module):
	"""Transformer Decoder Block (GPT-style)
	
	Architecture:
		x -> LayerNorm -> CausalSelfAttention -> Residual
		  -> LayerNorm -> MLP -> Residual
	
	Args:
		embed_dim: 埋め込み次元
		n_heads: アテンションヘッド数
		mlp_ratio: MLPの隠れ層サイズ比率
		dropout: ドロップアウト率
	"""
	def __init__(self, embed_dim, n_heads, mlp_ratio=4.0, dropout=0.1):
		super().__init__()
		self.ln1 = nn.LayerNorm(embed_dim)
		self.attn = CausalSelfAttention(embed_dim, n_heads, dropout)
		self.ln2 = nn.LayerNorm(embed_dim)
		
		mlp_hidden = int(embed_dim * mlp_ratio)
		self.mlp = nn.Sequential(
			nn.Linear(embed_dim, mlp_hidden),
			nn.GELU(),
			nn.Linear(mlp_hidden, embed_dim),
			nn.Dropout(dropout),
		)
		
	def forward(self, x, use_cache=False, past_kv=None):
		"""
		Args:
			x: (batch, seq_len, embed_dim)
			use_cache: K-Vキャッシュを使用
			past_kv: 過去のK, V
			
		Returns:
			out: (batch, seq_len, embed_dim)
			cache: (K, V) for next step
		"""
		# Self-attention with residual
		attn_out, cache = self.attn(self.ln1(x), use_cache, past_kv)
		x = x + attn_out
		
		# MLP with residual
		x = x + self.mlp(self.ln2(x))
		
		return x, cache


class PositionalEncoding(nn.Module):
	"""Sinusoidal Positional Encoding (Attention is All You Need)
	
	Args:
		embed_dim: 埋め込み次元
		max_len: 最大シーケンス長
		dropout: ドロップアウト率
	"""
	def __init__(self, embed_dim, max_len=5000, dropout=0.0):
		super().__init__()
		self.dropout = nn.Dropout(dropout)
		
		# Positional encoding行列を計算
		pe = torch.zeros(max_len, embed_dim)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(
			torch.arange(0, embed_dim, 2, dtype=torch.float) * 
			(-log(10000.0) / embed_dim)
		)
		
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
		
		self.register_buffer('pe', pe)
		
	def forward(self, x):
		"""
		Args:
			x: (batch, seq_len, embed_dim)
			
		Returns:
			out: (batch, seq_len, embed_dim)
		"""
		x = x + self.pe[:, :x.size(1)]
		return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
	"""学習可能なPositional Encoding
	
	Args:
		embed_dim: 埋め込み次元
		max_len: 最大シーケンス長
		dropout: ドロップアウト率
	"""
	def __init__(self, embed_dim, max_len=5000, dropout=0.0):
		super().__init__()
		self.dropout = nn.Dropout(dropout)
		self.pe = nn.Parameter(torch.zeros(1, max_len, embed_dim))
		nn.init.normal_(self.pe, std=0.02)
		
	def forward(self, x):
		"""
		Args:
			x: (batch, seq_len, embed_dim)
			
		Returns:
			out: (batch, seq_len, embed_dim)
		"""
		x = x + self.pe[:, :x.size(1)]
		return self.dropout(x)


def test_causal_attention():
	"""CausalSelfAttentionのユニットテスト"""
	print("Testing CausalSelfAttention...")
	
	batch_size = 4
	seq_len = 10
	embed_dim = 64
	n_heads = 4
	
	attn = CausalSelfAttention(embed_dim, n_heads)
	x = torch.randn(batch_size, seq_len, embed_dim)
	
	# 通常の forward
	out, _ = attn(x, use_cache=False)
	assert out.shape == (batch_size, seq_len, embed_dim), f"Expected {(batch_size, seq_len, embed_dim)}, got {out.shape}"
	
	# K-Vキャッシュ付き forward
	out1, cache1 = attn(x[:, :5], use_cache=True)
	out2, cache2 = attn(x[:, 5:6], use_cache=True, past_kv=cache1)
	assert out1.shape == (batch_size, 5, embed_dim)
	assert out2.shape == (batch_size, 1, embed_dim)
	
	print("✓ CausalSelfAttention tests passed")


def test_transformer_block():
	"""TransformerBlockのユニットテスト"""
	print("Testing TransformerBlock...")
	
	batch_size = 4
	seq_len = 10
	embed_dim = 64
	n_heads = 4
	
	block = TransformerBlock(embed_dim, n_heads)
	x = torch.randn(batch_size, seq_len, embed_dim)
	
	out, _ = block(x, use_cache=False)
	assert out.shape == (batch_size, seq_len, embed_dim)
	
	print("✓ TransformerBlock tests passed")


def test_positional_encoding():
	"""PositionalEncodingのユニットテスト"""
	print("Testing PositionalEncoding...")
	
	batch_size = 4
	seq_len = 10
	embed_dim = 64
	
	pe = PositionalEncoding(embed_dim)
	x = torch.randn(batch_size, seq_len, embed_dim)
	
	out = pe(x)
	assert out.shape == (batch_size, seq_len, embed_dim)
	
	# 学習可能版もテスト
	lpe = LearnablePositionalEncoding(embed_dim)
	out2 = lpe(x)
	assert out2.shape == (batch_size, seq_len, embed_dim)
	
	print("✓ PositionalEncoding tests passed")


if __name__ == "__main__":
	print("Running Transformer layers tests...\n")
	test_causal_attention()
	test_transformer_block()
	test_positional_encoding()
	print("\n✅ All tests passed!")

