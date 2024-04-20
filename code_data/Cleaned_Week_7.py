#!/usr/bin/env python
# coding: utf-8



import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy


# The Multi-Head Attention mechanism computes the attention between each pair of positions in a sequence. It consists of multiple “attention heads” that capture different aspects of the input sequence.
# 
# The MultiHeadAttention code initializes the module with input parameters and linear transformation layers. It calculates attention scores, reshapes the input tensor into multiple heads, and combines the attention outputs from all heads. The forward method computes the multi-head self-attention, allowing the model to focus on some different aspects of the input sequence.



class MultiHeadAttention(nn.Module):
    
    """
    The init constructor checks whether the provided d_model is divisible by the number of heads (num_heads). 
    It sets up the necessary parameters and creates linear transformations for
    query(W_q), key(W_k) and output(W_o) projections
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    """
     The scaled_dot_product_attention function computes the scaled dot-product attention given the 
     query (Q), key (K), and value (V) matrices. It uses the scaled dot product formula, applies a mask if 
     provided, and computes the attention probabilities using the softmax function.
    """    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
    
    """
    The split_heads and combine_heads functions handle the splitting and combining of the attention heads.
    They reshape the input tensor to allow parallel processing of different attention heads.
    """
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    """
     The forward function takes input query (Q), key (K), and value (V) tensors, 
     applies linear transformations, splits them into multiple heads, performs scaled dot-product attention,
     combines the attention heads, and applies a final linear transformation.
    """    
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


# The PositionWiseFeedForward class extends PyTorch’s nn.Module and implements a position-wise feed-forward network. The class initializes with two linear transformation layers and a ReLU activation function. The forward method applies these transformations and activation function sequentially to compute the output. This process enables the model to consider the position of input elements while making predictions.



class PositionWiseFeedForward(nn.Module):
    """
    PositionWiseFeedForward module. It takes d_model as the input dimension and d_ff 
    as the hidden layer dimension. 
    Two linear layers (fc1 and fc2) are defined with ReLU activation in between.
    """
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    """
    The forward function takes an input tensor x, applies the first linear transformation (fc1), 
    applies the ReLU activation, and then applies the second linear transformation (fc2). 
    The output is the result of the second linear transformation.
    """
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# Positional Encoding is used to inject the position information of each token in the input sequence. It uses sine and cosine functions of different frequencies to generate the positional encoding.
# 
# The PositionalEncoding class initializes with input parameters d_model and max_seq_length, creating a tensor to store positional encoding values. The class calculates sine and cosine values for even and odd indices, respectively, based on the scaling factor div_term. The forward method computes the positional encoding by adding the stored positional encoding values to the input tensor, allowing the model to capture the position information of the input sequence.



class PositionalEncoding(nn.Module):
    """
    The constructor (__init__) initializes the PositionalEncoding module. 
    It takes d_model as the dimension of the model and max_seq_length as the maximum sequence length. 
    It computes the positional encoding matrix (pe) using sine and cosine functions.
    """
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    """
    The forward function takes an input tensor x and adds the positional encoding to it. 
    The positional encoding is truncated to match the length of the input sequence (x.size(1)).
    """    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# An Encoder layer consists of a Multi-Head Attention layer, a Position-wise Feed-Forward layer, and two Layer Normalization layers.
# 
# The EncoderLayer class initializes with input parameters and components, including a MultiHeadAttention module, a PositionWiseFeedForward module, two layer normalization modules, and a dropout layer. The forward methods computes the encoder layer output by applying self-attention, adding the attention output to the input tensor, and normalizing the result. Then, it computes the position-wise feed-forward output, combines it with the normalized self-attention output, and normalizes the final result before returning the processed tensor.



class EncoderLayer(nn.Module):
    
    """
    The constructor (__init__) initializes the EncoderLayer module. 
    It takes hyperparameters such as d_model (model dimension), num_heads (number of attention heads), 
    d_ff (dimension of the feedforward network), and dropout (dropout rate). 
    It creates instances of MultiHeadAttention, PositionWiseFeedForward, and nn.LayerNorm. 
    Dropout is also defined as a module.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    """
    The forward function takes an input tensor x and a mask. 
    It applies the self-attention mechanism (self.self_attn), adds the residual connection 
    with layer normalization, applies the position-wise feedforward network (self.feed_forward),
    and again adds the residual connection with layer normalization. 
    Dropout is applied at both the self-attention and feedforward stages.
    The mask parameter is used to mask certain positions during the self-attention step, 
    typically to prevent attending to future positions in a sequence.
    """
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


# A Decoder layer consists of two Multi-Head Attention layers, a Position-wise Feed-Forward layer, and three Layer Normalization layers.
# 
# The DecoderLayer initializes with input parameters and components such as MultiHeadAttention modules for masked self-attention and cross-attention, a PositionWiseFeedForward module, three layer normalization modules, and a dropout layer.



class DecoderLayer(nn.Module):
    """
    The constructor (__init__) initializes the DecoderLayer module. 
    It takes hyperparameters such as d_model (model dimension), num_heads (number of attention heads), 
    d_ff (dimension of the feedforward network), and dropout (dropout rate). 
    It creates instances of MultiHeadAttention for both self-attention (self.self_attn) and cross-attention
    (self.cross_attn), PositionWiseFeedForward, and nn.LayerNorm. Dropout is also defined as a module.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    """
    The forward function takes an input tensor x, the output from the encoder (enc_output), 
    and masks for the source (src_mask) and target (tgt_mask). It applies the self-attention mechanism, 
    adds the residual connection with layer normalization, applies the cross-attention mechanism with the 
    encoder's output, adds another residual connection with layer normalization, applies the position-wise 
    feedforward network, and adds a final residual connection with layer normalization. 
    Dropout is applied at each stage.
    """
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x




class Transformer(nn.Module):
    """
    The constructor (__init__) initializes the Transformer module. 
    It takes several hyperparameters, including vocabulary sizes for the source and target languages 
    (src_vocab_size and tgt_vocab_size), model dimension (d_model), number of attention heads (num_heads), 
    number of layers (num_layers), dimension of the feedforward network (d_ff), maximum sequence length 
    (max_seq_length), and dropout rate (dropout).
    It sets up embeddings for both the encoder and decoder (encoder_embedding and decoder_embedding), 
    a positional encoding module (positional_encoding), encoder layers (encoder_layers), 
    decoder layers (decoder_layers), a linear layer (fc), and dropout.
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    """
     The generate_mask function creates masks for the source and target sequences. 
     It generates a source mask by checking if the source sequence elements are not equal to 0. 
     For the target sequence, it creates a mask by checking if the target sequence elements are not equal 
     to 0 and applies a no-peek mask to prevent attending to future positions.
    """
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    """
    The forward function takes source (src) and target (tgt) sequences as input. 
    It generates source and target masks using the generate_mask function. 
    The source and target embeddings are obtained by applying dropout to the positional embeddings of the 
    encoder and decoder embeddings, respectively. 
    The encoder layers are then applied to the source embeddings to get the encoder output (enc_output). 
    The decoder layers are applied to the target embeddings along with the encoder output, source mask, 
    and target mask to get the final decoder output (dec_output). The output is obtained by applying a linear layer to the decoder output.
    """
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output


# In this example, we will create a toy dataset for demonstration purposes. In practice, you would use a larger dataset, preprocess the text, and create vocabulary mappings for source and target languages.



src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# Generate random sample data
src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)


# We then train the model on the toy dataset.



criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

for epoch in range(100):
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")






