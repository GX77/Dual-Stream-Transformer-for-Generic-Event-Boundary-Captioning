import sys
import json
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
import logging
import os
import clip
from .vidswin import VideoSwinTransformerBackbone
logger = logging.getLogger(__name__)


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        self.log_softmax = nn.LogSoftmax(dim=-1)

        smoothing_value = label_smoothing / (tgt_vocab_size - 1)  # count for the ground-truth word
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        # one_hot[self.ignore_index] = 0
        self.register_buffer("one_hot", one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size, with indices in [-1, tgt_vocab_size-1], `-1` is ignored
        """
        valid_indices = target != self.ignore_index  # ignore examples with target value -1
        target = target[valid_indices]
        output = self.log_softmax(output[valid_indices])

        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        return F.kl_div(output, model_prob, reduction="sum")


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT"s gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionEncoding(nn.Module):
    """
    Add positional information to input tensor.
    :Examples:
        >>> model = PositionEncoding(d_model=6, max_len=10, dropout=0)
        >>> test_input1 = torch.zeros(3, 10, 6)
        >>> output1 = model(test_input1)
        >>> output1.size()
        >>> test_input2 = torch.zeros(5, 3, 9, 6)
        >>> output2 = model(test_input2)
        >>> output2.size()
    """

    def __init__(self, n_filters=128, max_len=500):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        """
        super(PositionEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_filters)  # (L, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_filters, 2).float() * - (math.log(10000.0) / n_filters))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # buffer is a tensor, not a variable, (L, D)

    def forward(self, x):
        """
        :Input: (*, L, D)
        :Output: (*, L, D) the same size as input
        """
        pe = self.pe.data[:x.size(-2), :]  # (#x.size(-2), n_filters)
        extra_dim = len(x.size()) - 2
        for _ in range(extra_dim):
            pe = pe.unsqueeze(0)
        x = x + pe
        return x


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query_states, key_states, value_states, attention_mask):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:

        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:

        """
        self_output = self.self(input_tensor, input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


def make_shifted_mask(input_mask, max_v_len, max_t_len, memory_len=0):
    """
    Args:
        input_mask: (N, L) with `1` indicates valid bits, `0` indicates pad
        max_v_len: int, the first `max_v_len` is for video and its padding, the length
            of the rest of the bits is `max_t_len`. We have L = `max_v_len` + `max_t_len`.
            Note max_v_len may also include the memory len (M), thus max_v_len += M
        max_t_len: int
        memory_len: int, M
    Returns:

    >>> max_v_len = 2; max_t_len=3; input_mask = torch.randn(2, 5)
    >>> make_pad_shifted_mask(input_mask, max_v_len, max_t_len)[0]
    tensor([[1., 1., 0., 0., 0.],
            [1., 1., 0., 0., 0.],
            [1., 1., 1., 0., 0.],
            [1., 1., 1., 1., 0.],
            [1., 1., 1., 1., 1.]])
    """
    bsz, seq_len = input_mask.shape
    assert max_v_len + max_t_len + memory_len == seq_len
    shifted_mask = input_mask.new_zeros(bsz, max_v_len + max_t_len, seq_len)  # (N, L, M+L)
    shifted_mask[:, :, :memory_len + max_v_len] = 1
    shifted_mask[:, max_v_len:, memory_len + max_v_len:] = torch.tril(input_mask.new_ones(max_t_len, max_t_len),
                                                                      diagonal=0)
    return shifted_mask


def make_pad_shifted_mask(input_mask, max_v_len, max_t_len, n, memory_len=0):
    """input_mask: (N, L), """
    shifted_mask = make_shifted_mask(input_mask, max_v_len, max_t_len, memory_len=memory_len)
    masks = shifted_mask * input_mask.unsqueeze(1)
    return masks


def make_video_only_mask(input_mask, max_v_len):
    video_only_mask = copy.deepcopy(input_mask)
    video_only_mask[:, max_v_len:] = 0
    return video_only_mask


class BertLayerNoMemory(nn.Module):
    def __init__(self, config):
        super(BertLayerNoMemory, self).__init__()
        self.config = config
        self.attention = BertAttention(config)
        self.hidden_intermediate = BertIntermediate(config)
        self.memory_intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.region_num = config.region_num

    def forward(self, hidden_states, attention_mask, m):
        """
        Args:
            hidden_states: (N, L, D)
            attention_mask: (N, L)
        Returns:

        """
        max_v_len, max_t_len = m, self.config.max_t_len
        # self-attention, need to shift right
        shifted_self_mask = make_pad_shifted_mask(attention_mask, max_v_len, max_t_len, self.region_num, memory_len=0)  # (N, L, L)
        attention_output = self.attention(hidden_states, shifted_self_mask)  # (N, L, D)
        intermediate_output = self.hidden_intermediate(attention_output)  # (N, L, D)
        layer_output = self.output(intermediate_output, attention_output)  # (N, L, D)
        return layer_output


def make_pad_cross_mask_1(q_mask, v_mask, m_q, m_v, n):
    max_v_len = 30
    max_r_len = 20 * n + 2
    max_asr_len = 5

    m = torch.zeros(q_mask.size(0), m_q, m_v).cuda()
    t = v_mask.unsqueeze(1).expand(v_mask.size(0), m_q, v_mask.size(-1))[..., :max_r_len + max_asr_len].cuda()
    m[:, :, :max_r_len + max_asr_len] = t
    m[:, max_v_len + max_asr_len:max_v_len + max_asr_len + 22,
    max_r_len + max_asr_len:max_r_len + max_asr_len + 22] = torch.tril(torch.ones(22, 22), diagonal=0)
    m = m * q_mask.unsqueeze(-1)
    return m


def make_pad_cross_mask_2(q_mask, v_mask, m_q, m_v, n):
    max_v_len = 30
    max_r_len = 20 * n + 2
    max_asr_len = 5
    m = torch.zeros(q_mask.size(0), m_q, m_v).cuda()
    t = v_mask.unsqueeze(1).expand(v_mask.size(0), m_q, v_mask.size(-1))[..., :max_v_len + max_asr_len].cuda()
    m[:, :, :max_v_len + max_asr_len] = t
    m[:, max_r_len + max_asr_len:max_r_len + max_asr_len + 22,
    max_v_len + max_asr_len:max_v_len + max_asr_len + 22] = torch.tril(torch.ones(22, 22), diagonal=0)
    m = m * q_mask.unsqueeze(-1)
    return m


class BertAttention_cross(nn.Module):
    def __init__(self, config):
        super(BertAttention_cross, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, q, kv, attention_mask):
        self_output = self.self(q, kv, kv, attention_mask)
        attention_output = self.output(self_output, q)
        return attention_output


class BertLayerNoMemory_Cross(nn.Module):
    def __init__(self, config):
        super(BertLayerNoMemory_Cross, self).__init__()
        self.config = config
        self.attention = BertAttention_cross(config)
        self.hidden_intermediate = BertIntermediate(config)
        self.memory_intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.region_num = config.region_num

    def forward(self, q, kv, q_mask, kv_mask, m):
        # self-attention, need to shift right
        max_v_len = 30
        max_asr_len = 5
        if m == max_v_len + max_asr_len:
            shifted_self_mask = make_pad_cross_mask_1(q_mask, kv_mask, m_q=q_mask.size(-1), m_v=kv_mask.size(-1),
                                                      n=self.region_num)  # (N, L, L)
        else:
            shifted_self_mask = make_pad_cross_mask_2(q_mask, kv_mask, m_q=q_mask.size(-1), m_v=kv_mask.size(-1),
                                                      n=self.region_num)
        attention_output = self.attention(q, kv, shifted_self_mask)  # (N, L, D)
        intermediate_output = self.hidden_intermediate(attention_output)  # (N, L, D)
        layer_output = self.output(intermediate_output, attention_output)  # (N, L, D)
        return layer_output


class BertEncoderNoMemory(nn.Module):
    def __init__(self, config):
        super(BertEncoderNoMemory, self).__init__()
        self.layer_sa = nn.ModuleList([BertLayerNoMemory(config) for _ in range(6)])
        self.layer_ca = nn.ModuleList([BertLayerNoMemory_Cross(config) for _ in range(6)])
        self.max_v_len = config.max_v_len
        self.region_num = config.region_num

    def forward(self, hidden_states, hidden_states2, attention_mask, attention_mask2, output_all_encoded_layers=True):
        n = self.region_num
        max_r_len = 20 * n + 2
        max_asr_len = 5
        for i in [0, 1, 2]:
            hidden_states = self.layer_sa[i](hidden_states, attention_mask, m=self.max_v_len + max_asr_len)
            hidden_states2 = self.layer_sa[i + 3](hidden_states2, attention_mask2, m=max_r_len + max_asr_len)  # 左

            video = self.layer_ca[i](hidden_states, hidden_states2, attention_mask, attention_mask2,
                                     m=self.max_v_len + max_asr_len)  # 右
            region = self.layer_ca[i + 3](hidden_states2, hidden_states, attention_mask2, attention_mask,
                                          m=max_r_len + max_asr_len)
            hidden_states = video
            hidden_states2 = region

        return hidden_states, hidden_states2


class BertEmbeddingsWithVideo(nn.Module):
    """Construct the embeddings from word (+ video), position and token_type embeddings.
    input_ids (batch_size, sequence_length), with [1, sequence_length_1 + 1] filled with [VID]
    video_features (batch_size, sequence_length),
    with [1, sequence_length_1 + 1] as real features, others as zeros
    ==> video features and word embeddings are merged together by summing up.
    """

    def __init__(self, config, add_postion_embeddings=True):
        super(BertEmbeddingsWithVideo, self).__init__()
        """add_postion_embeddings: whether to add absolute positional embeddings"""
        self.add_postion_embeddings = add_postion_embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.word_vec_size, padding_idx=0)  # 词嵌入
        self.word_embeddings2 = nn.Embedding(config.vocab_size, config.word_vec_size, padding_idx=0)
        self.word_fc = nn.Sequential(  # 300->768
            BertLayerNorm(config.word_vec_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.word_vec_size, config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.word_fc2 = nn.Sequential(  # 300->768
            BertLayerNorm(config.word_vec_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.word_vec_size, config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.video_embeddings = nn.Sequential(  # 3072->768
            BertLayerNorm(config.video_feature_size, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.video_feature_size, config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.region_embeddings = nn.Sequential(  # 3072->768
            BertLayerNorm(2048, eps=config.layer_norm_eps),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(2048, config.hidden_size),
            nn.ReLU(True),
            BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )

        if self.add_postion_embeddings:
            self.position_embeddings = PositionEncoding(n_filters=config.hidden_size,
                                                        max_len=config.max_position_embeddings + 2500)
            self.position_embeddings2 = PositionEncoding(n_filters=config.hidden_size, max_len=2000)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.token_type_embeddings2 = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm2 = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def set_pretrained_embedding(self, pretrained_embedding, freeze=True):
        """Note the from_pretrained does not work in-place, so you need to assign value to the embedding"""
        assert pretrained_embedding.shape == self.word_embeddings.weight.shape  # ensure equal shape
        self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze,
                                                            padding_idx=self.word_embeddings.padding_idx)
        self.word_embeddings2 = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze,
                                                             padding_idx=self.word_embeddings.padding_idx)

    def forward(self, input_ids, input_ids2, video_features, region_features, token_type_ids, token_type_ids2):
        """
        Args:
            input_ids: (N, L)
            video_features: (N, L, D)
            token_type_ids: (N, L, D)

        Returns:

        """
        words_embeddings = self.word_fc(self.word_embeddings(input_ids))
        video_embeddings = self.video_embeddings(video_features)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings1 = words_embeddings + video_embeddings + token_type_embeddings
        if self.add_postion_embeddings:
            embeddings1 = self.position_embeddings(embeddings1)
        embeddings1 = self.LayerNorm(embeddings1)
        embeddings1 = self.dropout(embeddings1)

        words_embeddings2 = self.word_fc2(self.word_embeddings2(input_ids2))
        region_embeddings = self.region_embeddings(region_features)
        token_type_embeddings2 = self.token_type_embeddings2(token_type_ids2)
        embeddings2 = words_embeddings2 + region_embeddings + token_type_embeddings2
        if self.add_postion_embeddings:
            embeddings2 = self.position_embeddings2(embeddings2)
        embeddings2 = self.LayerNorm2(embeddings2)
        embeddings2 = self.dropout2(embeddings2)

        return embeddings1, embeddings2  # (N, L, D)


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = gelu
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        """(N, L, D)"""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights=None):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        if config.share_wd_cls_weight:
            assert bert_model_embedding_weights is not None, \
                "bert_model_embedding_weights should not be None " \
                "when setting --share_wd_cls_weight flag to be true"
            assert config.hidden_size == bert_model_embedding_weights.size(1), \
                "hidden size has be the same as word embedding size when " \
                "sharing word embedding weight and classifier weight"
            self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                     bert_model_embedding_weights.size(0),
                                     bias=False)
            self.decoder.weight = bert_model_embedding_weights
        else:
            self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        """(N, L, D)"""
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states  # (N, L, vocab_size)

configs = {
    'video_swin_t_p4w7':
        dict(patch_size=(2, 4, 4),
             embed_dim=96,
             depths=[2, 2, 6, 2],
             num_heads=[3, 6, 12, 24],
             window_size=(8, 7, 7),
             mlp_ratio=4.,
             qkv_bias=True,
             qk_scale=None,
             drop_rate=0.,
             attn_drop_rate=0.,
             drop_path_rate=0.2,
             patch_norm=True,
             use_checkpoint=False
             ),
    'video_swin_s_p4w7':
        dict(patch_size=(2, 4, 4),
             embed_dim=96,
             depths=[2, 2, 18, 2],
             num_heads=[3, 6, 12, 24],
             window_size=(8, 7, 7),
             mlp_ratio=4.,
             qkv_bias=True,
             qk_scale=None,
             drop_rate=0.,
             attn_drop_rate=0.,
             drop_path_rate=0.2,
             patch_norm=True,
             use_checkpoint=False
             ),
    'video_swin_b_p4w7':
        dict(patch_size=(2, 4, 4),
             embed_dim=128,
             depths=[2, 2, 18, 2],
             num_heads=[4, 8, 16, 32],
             window_size=(8, 7, 7),
             mlp_ratio=4.,
             qkv_bias=True,
             qk_scale=None,
             drop_rate=0.,
             attn_drop_rate=0.,
             drop_path_rate=0.2,
             patch_norm=True,
             use_checkpoint=False
             )
}

class NonRecurTransformer(nn.Module):
    def __init__(self, config):
        super(NonRecurTransformer, self).__init__()
        self.config = config
        self.region_num = config.region_num
        self.embeddings = BertEmbeddingsWithVideo(config, add_postion_embeddings=True)
        self.encoder = BertEncoderNoMemory(config)
        decoder_classifier_weight = self.embeddings.word_embeddings.weight \
            if self.config.share_wd_cls_weight else None  ###########
        self.decoder = BertLMPredictionHead(config, decoder_classifier_weight)
        self.decoder2 = BertLMPredictionHead(config, decoder_classifier_weight)
        self.loss_func = LabelSmoothingLoss(config.label_smoothing, config.vocab_size,
                                            ignore_index=-1) if "label_smoothing" in config and config.label_smoothing > 0 else nn.CrossEntropyLoss(
            ignore_index=-1)
        self.loss_func2 = LabelSmoothingLoss(config.label_smoothing, config.vocab_size,
                                             ignore_index=-1) if "label_smoothing" in config and config.label_smoothing > 0 else nn.CrossEntropyLoss(
            ignore_index=-1)
        self.apply(self.init_bert_weights)

        cfgs = configs['video_swin_t_p4w7']
        self.extr_3d = VideoSwinTransformerBackbone(True, './swin_tiny_patch244_window877_kinetics400_1k.pth', True, **cfgs)

        self.extr_2d, _ = clip.load("ViT-B/32")


    def init_bert_weights(self, module):
        """ Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    # def forward(self, input_ids, video_features, region_features,input_masks, token_type_ids, input_labels,tags):
    def forward(self, input_ids, input_ids2, vf_rgb, region_features, input_masks, input_masks2, token_type_ids,
                token_type_ids2, input_labels):
        """
        Args:
            input_ids: [(N, L)]
            video_features: [(N, L, D_v)] * step_size
            input_masks: [(N, L)] * step_size with 1 indicates valid bits
            token_type_ids: [(N, L)] * step_size, with `0` on the first `max_v_len` bits, `1` on the last `max_t_len`
            input_labels: [(N, L)] * step_size, with `-1` on ignored positions
        """
        n = self.region_num
        max_r_len = 20 * n + 2
        max_asr_len = 5
        out_3d = self.extr_3d(vf_rgb)
        vf_rgb = vf_rgb[:, ::2].reshape(-1, vf_rgb.size(-3), vf_rgb.size(-2), vf_rgb.size(-1))
        out_2d = self.extr_2d.encode_image(vf_rgb).reshape(out_3d.size(0), out_3d.size(1), -1)
        out = torch.cat((out_2d, out_3d), dim=-1)
        video_features = torch.cat((torch.zeros(out.shape[0], 1, out.size(-1)).cuda(), out,
                                    torch.zeros(out.shape[0], input_masks.size(1) - 1 - out.shape[1],
                                                out.size(-1)).cuda()), dim=1)

        embeddings1, embeddings2 = self.embeddings(input_ids, input_ids2, video_features, region_features,
                                                   token_type_ids, token_type_ids2)  # (N, L, D)

        encoded_v, encoded_r = self.encoder(embeddings1, embeddings2, input_masks, input_masks2, output_all_encoded_layers=False)

        prediction_scores_v = self.decoder(encoded_v)[:, -22:]  # (N, L, vocab_size)
        prediction_scores_r = self.decoder2(encoded_r)[:, max_r_len + max_asr_len:max_r_len + max_asr_len + 22]
        if input_labels is not None:
            caption_loss = self.loss_func(prediction_scores_v.contiguous().view(-1, self.config.vocab_size),
                                          input_labels.view(-1))
            region_loss = self.loss_func2(prediction_scores_r.contiguous().view(-1, self.config.vocab_size),
                                          input_labels.view(-1))
            loss = region_loss + caption_loss
        else:
            loss = None
        return loss, prediction_scores_r


# remind me of what the configs are
base_config = edict(
    hidden_size=768,
    vocab_size=None,  # get from word2idx
    video_feature_size=2048,
    max_position_embeddings=None,  # get from max_seq_len
    max_v_len=30,  # max length of the videos
    max_t_len=30,  # max length of the text
    n_memory_cells=10,  # memory size will be (n_memory_cells, D)
    type_vocab_size=2,
    layer_norm_eps=1e-12,  # bert layernorm
    hidden_dropout_prob=0.1,  # applies everywhere except attention
    num_hidden_layers=2,  # number of transformer layers
    attention_probs_dropout_prob=0.1,  # applies only to self attention
    intermediate_size=768,  # after each self attention
    num_attention_heads=12,
    memory_dropout_prob=0.1
)
