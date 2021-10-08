# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""
 
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
 
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
 
# -*- coding: utf-8 -*-
# -*- created_time: 2020/12/4 -*-
#!/usr/bin/env Python
# coding=utf-8
 
# 利用sys.path.append()将路径D:/Python/python_virtual_environment/machine-learning/Lib/site-packages/transformers与
# 路径D:/Python/python_virtual_environment/machine-learning/Lib/site-packages/transformers/models/gpt2添加入python的搜索模块的路径集,
# 这样在引入下方activations、file_utils、modeling_outputs、modeling_utils、utils与configuration_gpt2这几个模块时便不会报错。
# 因为此脚本文件原本和这几个模块在同一目录下, 但此时将这个脚本文件单独拷贝出来做注释之后便不在原先的目录之下了，
# 因此需要将原先的目录路径加入python的搜索模块的路径集中, 以确保在下方引入这六个模块时不会出错。
import sys
sys.path.append("D:/Python/python_virtual_environment/machine-learning/Lib/site-packages/transformers")
sys.path.append("D:/Python/python_virtual_environment/machine-learning/Lib/site-packages/transformers/models/gpt2")
 
from ...activations import ACT2FN
from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithPastAndCrossAttentions,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import (
    Conv1D,
    PreTrainedModel,
    SequenceSummary,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)
from ...utils import logging
from .configuration_gpt2 import GPT2Config
 
 
logger = logging.get_logger(__name__)
 
_CONFIG_FOR_DOC = "GPT2Config"
_TOKENIZER_FOR_DOC = "GPT2Tokenizer"
 
GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "distilgpt2",
    # See all GPT-2 models at https://huggingface.co/models?filter=gpt2
]
 
 
def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import re
 
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())
 
    for name, array in zip(names, arrays):
        name = name[6:]  # skip "model/"
        name = name.split("/")
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model
 
 
class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False, is_cross_attention=False):
        super().__init__()
 
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        # 利用断言函数判断此时隐藏状态的维度数n_state除以注意力头数config.n_head之后是否能整除.
        assert n_state % config.n_head == 0
 
        # 下方的self.register_buffer()函数的操作相当于创建了两个Attention类中的self属性, 即为self.bias属性
        # 与self.masked_bias属性；
        # 其中self.bias属性为一个下三角矩阵(对角线下元素全为1, 对角线上元素全为0), 其形状为(1, 1, n_ctx, n_ctx),
        # 也即形状相当于(1, 1, 1024, 1024)；
        # 而self.masked_bias属性则为一个极大的负数-1e4；
        self.register_buffer(
            "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
 
 
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
 
        self.is_cross_attention = is_cross_attention
        if self.is_cross_attention:
            # self.c_attn = Conv1D(2 * n_state, nx)相当于全连接层, 其将输入张量的最后一个维度的维度数由nx(768)投影为
            # 2 * n_state(2*768), 此时n_state = nx = num_head*head_features = 768.
            self.c_attn = Conv1D(2 * n_state, nx)
 
            # self.q_attn = Conv1D(n_state, nx)相当于全连接层, 其将输入张量的最后一个维度的维度数由nx(768)投影为
            # n_state(768), 此时n_state = nx = num_head*head_features = 768.
            self.q_attn = Conv1D(n_state, nx)
 
        else:
            # self.c_attn = Conv1D(3 * n_state, nx)相当于全连接层, 其将输入张量的最后一个维度的维度数由nx(768)投影为
            # 2 * n_state(2*768), 此时n_state = nx = num_head*head_features = 768.
            self.c_attn = Conv1D(3 * n_state, nx)
 
        # 此处self.c_proj()为Conv1D(n_state, nx)函数(all_head_size=n_state=nx=768), 相当于一个全连接层的作用,
        # 其将此时的多头注意力聚合操作结果张量a的最后一个维度all_head_size由n_state(768)的维度数投影为nx(768)的维度数.
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()
 
 
    # prune_heads()可结合 https://github.com/huggingface/transformers/issues/850 理解.
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_head, self.split_size // self.n_head, self.pruned_heads
        )
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])
 
        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)
 
        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)
 
 
    def merge_heads(self, x):
        # 此时x为: 利用计算得到的注意力分数张量对value张量进行注意力聚合后得到的注意力结果张量.
        # x的形状为(batch_size, num_head, sql_len, head_features).
 
        # 此时先将注意力结果张量x的形状变为(batch_size, sql_len, num_head, head_features)
        x = x.permute(0, 2, 1, 3).contiguous()
        # new_x_shape为(batch_size, sql_len, num_head*head_features) =》(batch_size, sql_len, all_head_size)
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
 
        # 此时将注意力结果张量x的注意力头维度num_head与注意力特征维度head_features进行合并变为all_head_size维度,
        # 注意力结果张量x的形状变为(batch_size, sql_len, all_head_size).
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states， (batch_size, sql_len, all_head_size).
 
 
    def split_heads(self, x, k=False):
        # 此时new_x_shape为: (batch_size, sql_len, num_head, head_features)
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        # 将输入的张量x(可能为query、key、value张量)变形为: (batch_size, sql_len, num_head, head_features).
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
 
        # 若此时输入的张量为key张量,则需要将key张量再变形为(batch_size, num_head, head_features, sql_len).
        # 因为此时key张量需要以[query * key]的形式与query张量做内积运算, 因此key张量需要将head_features变换到第三维度,
        # 将sql_len变换到第四维度,这样[query * key]内积运算之后的注意力分数张量的形状才能符合(batch_size, num_head, sql_len, sql_len).
        if k:
            return x.permute(0, 2, 3, 1)  # (batch_size, num_head, head_features, sql_len)
 
        # 若此时输入的张量为query张量或value张量, 则将张量维度再变换为(batch_size, num_head, sql_len, head_features)即可,
        # 即将sql_len与num_head调换维度.
        else:
            return x.permute(0, 2, 1, 3)  # (batch_size, num_head, sql_len, head_features)
 
 
    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        ''''''
        '''
        此时query张量形状为: (batch_size, num_head, 1, head_features)
        key张量的形状为: (batch_size, num_head, head_features, sql_len+1)
        value张量的形状为: (batch_size, num_head, sql_len+1, head_features)
 
        此时key张量以[query * key]的形式与query张量做内积运算, key张量已在split_heads()操作与past_key合并操作中
        提前将head_features变换到第三维度, 将sql_len+1变换到第四维度,这样[query * key]内积运算之后的注意力分数张量w的
        形状才能符合(batch_size, num_head, 1, sql_len+1).
        '''
        w = torch.matmul(q, k)  # 注意力分数张量w: (batch_size, num_head, 1, sql_len+1)
 
        # 对注意力分数张量w中的值进行缩放(scaled), 缩放的除数为注意力头特征数head_features的开方值.
        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)
 
        # 此时nd与ns两个维度相当于1与seq_len+1
        nd, ns = w.size(-2), w.size(-1)
 
        # 此处的操作为利用torch.where(condition, x, y)函数,将注意力分数张量w在mask.bool()条件张量为True(1)的相同位置的值
        # 保留为w中的原值, 将在mask.bool()条件张量为True(0)的相同位置的值变为self.masked_bias(-1e4)的值.
        '''<1> GPT2Model第一次迭代时输入GPT2Model的forward()函数中的past_key_values参数为None, 此时nd与ns维度才会相等, 
        在nd与ns维度相等的情况下此操作的结果等价于让注意力分数张量w与attention_mask张量相加的结果。
        <2> 若为GPT2Mode第二次及之后的迭代时, nd与ns两个维度相当于1与seq_len+1, 此时对self.bias进行切片操作时, 
        ns - nd等于seq_len+1 - 1即结果为seq_len, 即此时切片操作相当于self.bias[:, :, seq_len : seq_len+1, :seq_len+1],
        此操作的意义在于对此次迭代中, 最新的token的注意力分数上添加GPT2中的下三角形式的注意力遮罩.'''
        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            # 此时self.bias属性为一个下三角矩阵(对角线下元素全为1, 对角线上元素全为0), 其形状为(1, 1, n_ctx, n_ctx),
            # 也即形状相当于(1, 1, 1024, 1024)；但此处对self.bias进行切片操作时, ns - nd等于seq_len+1 - 1即结果为seq_len,
            # 即此时切片操作相当于self.bias[:, :, seq_len : seq_len+1, :seq_len+1]。
            '''此时mask张量(经过大张量self.bias切片获得)的形状为(1, 1, 1, seq_len + 1).'''
            mask = self.bias[:, :, ns - nd: ns, :ns]
            '''此操作的意义在于对此次迭代中, 最新的token的注意力分数上添加GPT2中的下三角形式注意力遮罩.'''
            w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))
 
        # 让注意力分数张量w与attention_mask张量相加, 以达到让填充特殊符[PAD]处的注意力分数为一个很大的负值的目的,这样在下面将
        # 注意力分数张量w输入Softmax()层计算之后, 填充特殊符[PAD]处的注意力分数将会变为无限接近0的数, 以此让填充特殊符[PAD]
        # 处的注意力分数极小, 其embedding嵌入值基本不会在多头注意力聚合操作中被获取到.
        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask
 
        # 注意力分数张量w: (batch_size, num_head, 1, sql_len+1).
        # 将注意力分数张量w输入进Softmax()层中进行归一化计算, 计算得出最终的注意力分数,
        # 再将注意力分数张量w输入进Dropout层self.attn_dropout()中进行正则化操作, 防止过拟合.
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
 
        # Mask heads if we want to, 对注意力头num_head维度的mask操作.
        if head_mask is not None:
            w = w * head_mask
 
        # 多头注意力聚合操作: 注意力分数张量w与value张量进行内积
        # 注意力分数张量w形状: (batch_size, num_head, 1, sql_len+1)
        # value张量形状: (batch_size, num_head, sql_len+1, head_features)
        # 多头注意力聚合操作结果张量形状: (batch_size, num_head, 1, head_features), head_features=768.
        outputs = [torch.matmul(w, v)]
        # 若同时返回注意力分数张量w, 则将w张量添加入outputs列表中.
        if output_attentions:
            outputs.append(w)
 
        return outputs
 
 
    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        # <1> 此时的隐藏状态hidden_states的形状为 (batch_size, 1, nx), 此时nx = n_state = n_embed = head_features = 768，
        #     即此时隐藏状态hidden_states的形状为(batch_size, 1, 768)。
        # <2> 此时layer_past为一个存储着past_key张量与past_value张量的大张量, 其
        #     形状为(2, batch_size, num_head, sql_len, head_features).
        # <3> attention_mask张量为注意力遮罩张量, 其让填充特殊符[PAD]处的注意力分数极小,
        #     其embedding嵌入值基本不会在多头注意力聚合操作中被获取到.
 
        if encoder_hidden_states is not None:
            assert hasattr(
                self, "q_attn"
            ), "If class is used as cross attention, the weights `q_attn` have to be defined. " \
               "Please make sure to instantiate class with `Attention(..., is_cross_attention=True)`."
 
            '''self.crossattention()的Cross_Attention运算过程则是将LayerNormalization之后的hidden_states通过
            'self.q_attn = Conv1D(n_state, nx)(第168行代码)'将hidden_states的形状由(batch_size,1, 768)投影为(batch_size,1, 768),
            将此投影之后的hidden_states赋值作为query张量；
            再将此时从编码器(encoder)中传过来的编码器隐藏状态encoder_hidden_states通过'self.c_attn = Conv1D(2 * n_state, nx)
            (第164行代码)'将encoder_hidden_states的形状由(batch_size, enc_seq_len, 768)投影为(batch_size, enc_seq_len, 2 * 768),
            将投影后的encoder_hidden_states在在第三维度(dim=2)上拆分为两份分别赋为key、value,
            其形状都为(batch_size, enc_seq_len, 768)；  此时n_state = nx = num_head*head_features = 768.
             
            之后经过split_heads()函数拆分注意力头之后:
            query张量的形状变为(batch_size, num_head, 1, head_features),
            key张量的形状变为(batch_size, num_head, head_features, enc_seq_len),
            value张量的形状变为(batch_size, num_head, enc_seq_len, head_features).
             
            此时计算出的cross_attention张量形状为(batch_size, num_head, 1, enc_seq_len).'''
 
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
 
        else:
            '''此时隐藏状态hidden_states的形状为(batch_size, 1, 768), 将其输入进全连接层self.c_attn中后,
            其Conv1D(3 * n_state, nx)操作(nx=n_state=768)便会将hidden_states的第三维度数由 768维 投影为 3 * 768维,
            此时的hidden_states张量的形状为(batch_size, 1, 3 * 768), 最后将hidden_states张量在第三个维度(维度数3 * 768)上
            切分为三块, 将这切分出的三块各当成query, key, value张量, 则每个张量的形状都为(batch_size, 1, 768).
            此时n_state = nx = num_head*head_features = 768.
             
            之后经过split_heads()函数拆分注意力头且key、value张量分别与past_key、past_value张量合并之后:
            query张量的形状变为(batch_size, num_head, 1, head_features),
            key张量的形状变为(batch_size, num_head, head_features, sql_len+1),
            value张量的形状变为(batch_size, num_head, sql_len+1, head_features).'''
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
 
 
        '''第一次迭代时query、key、value张量的seq_len维度处的维度数就为seq_len而不是1, 第二次之后seq_len维度的维度数皆为1.'''
        # 此时经过'注意力头拆分函数split_heads()'之后的query、key、value三个张量的形状分别为:
        # query: (batch_size, num_head, 1, head_features)
        # key: (batch_size, num_head, head_features, 1)
        # value: (batch_size, num_head, 1, head_features)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
 
        if layer_past is not None:
            '''第一次迭代时query、key、value张量的seq_len维度处的维度数就为seq_len而不是1, 第二次之后seq_len维度的维度数皆为1.'''
            '''<1> 本次迭代中新的key张量
            此时需要通过layer_past[0].transpose(-2, -1)操作将past_key张量的形状变为(batch_size, num_head, head_features, sql_len),
            而此时key张量的形状为(batch_size, num_head, head_features, 1), 这样在下方就方便将past_key张量与key张量在最后
            一个维度(dim=-1)处进行合并, 这样就将当前token的key部分加入了past_key的seq_len中, 以方便模型在后面预测新的token,
            此时新的key张量的形状为: (batch_size, num_head, head_features, sql_len+1), new_seq_len为sql_len+1。
             <2> 本次迭代中新的value张量
            而此时past_value不用变形, 其形状为(batch_size, num_head, sql_len, head_features), 而此时value张量的形状为
            (batch_size, num_head, 1, head_features), 这样在下方就方便将past_value张量与value张量在倒数第二个
            维度(dim=-2)处进行合并, 这样就将当前token的value部分加入了past_value的seq_len中, 以方便模型在后面预测新的token,
            此时新的value张量的形状为: (batch_size, num_head, sql_len+1, head_features), new_seq_len为sql_len+1。
           '''
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
 
        # config对应的GPT2Config()类中的use_cache默认为True.但此时若为Cross_Attention运算过程, 则此时不会指定use_cache,
        # 而此时use_cache属性即为False(因为Attention类中use_cache属性默认为False, 除非指定config对应的GPT2Config()类
        # 中的use_cache属性其才会为True).
        if use_cache is True:
            # 若use_cache为True, 此时将key张量的最后一个维度与倒数第二个维度互换再与value张量进行stack合并,
            # 此时key.transpose(-2, -1)的形状为(batch_size, num_head, sql_len+1, head_features),
            # 此时torch.stack()操作后的present张量形状为(2, batch_size, num_head, sql_len+1, head_features)。
            '''present张量形状: (2, batch_size, num_head, sql_len+1, head_features),
            即present张量是用来存储此次迭代中的key张量与上一次迭代中的past_key张量(layer_past[0])合并、
            本次迭代的value张量与上一次迭代中的past_value张量(layer_past[1])合并后所得的新的key张量与value张量的.'''
            present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        else:
            present = (None,)
 
 
        '''此时query张量形状为: (batch_size, num_head, 1, head_features)
        key张量的形状为: (batch_size, num_head, head_features, sql_len+1)
        value张量的形状为: (batch_size, num_head, sql_len+1, head_features)'''
        # 若output_attentions为True, 则self._attn()函数返回的attn_outputs列表中的第二个值为注意力分数张量w.
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
 
 
        # 此时self._attn()函数返回的attn_outputs列表中的第一个元素为多头注意力聚合操作结果张量a,
        # a张量的形状为(batch_size, num_head, 1, head_features);
        # 若output_attentions为True, 则此时self._attn()函数返回的attn_outputs列表中的第二个元素为
        # 注意力分数张量w, 其形状为(batch_size, num_head, 1, seq_len + 1).
        a = attn_outputs[0]
 
        '''此时经过'多头注意力头合并函数self.merge_heads()'后的多头注意力聚合操作结果张量a的形状
        变为(batch_size, 1, all_head_size), 其中 all_head_size 等于 num_head * head_features, head_features=768.
        all_head_size维度的维度数为768,等于n_state,也等于nx, 即all_head_size=n_state=nx=768.'''
        a = self.merge_heads(a)
 
        # 此处self.c_proj()为Conv1D(n_state, nx)函数(all_head_size=n_state=nx=768), 相当于一个全连接层的作用,
        # 其将此时的多头注意力聚合操作结果张量a的最后一个维度all_head_size由n_state(768)的维度数投影为nx(768)的维度数.
        a = self.c_proj(a)
        a = self.resid_dropout(a)  # 残差dropout层进行正则化操作, 防止过拟合.
 
        # 此时多头注意力聚合操作结果张量a的形状为(batch_size, 1, all_head_size),
        # 其中 all_head_size 等于 num_head * head_features；all_head_size维度的维度数为768,
        # 等于n_state,也等于nx, 即all_head_size=n_state=nx=n_embed=768.
        outputs = [a, present] + attn_outputs[1:]
 
        # 此时返回的outputs列表中:
        # <1> 第一个值为多头注意力聚合操作结果张量a, 形状为(batch_size, 1, all_head_size), all_head_size=n_state=nx=n_embd=768.
        # <2> 第二个值为上方的present张量, 其存储着past_key张量与这次迭代的key张量合并后的新key张量, 以及
        #     past_value张量与这次迭代的value张量合并后的新value张量, 其形状为(2, batch_size, num_head, sql_len+1, head_features).
        # <3> 若output_attentions为True, 则第三个值为attn_outputs列表中的注意力分数张量w,
        #     其形状为(batch_size, num_head, 1, seq_len + 1).
        return outputs  # a, present, (attentions)
 
 
class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        # 此时nx=n_embed=768；
        # 而n_state实际为inner_dim，即n_state为4 * n_embd等于3072。
        nx = config.n_embd
 
        # self.c_fc = Conv1D(n_state, nx)相当于全连接层, 其将输入张量的最后一个维度的维度数由nx(768)投影为
        # n_state(3072), 此时n_state=3072.
        self.c_fc = Conv1D(n_state, nx)
        # self.c_proj = Conv1D(nx, n_state)相当于全连接层, 其将输入张量的最后一个维度的维度数由n_state(3072)投影为
        # nx(768), 此时n_state=3072.
        self.c_proj = Conv1D(nx, n_state)
 
        # 激活函数gelu.
        self.act = ACT2FN[config.activation_function]
        # 残差dropout层进行正则化操作, 防止过拟合.
        self.dropout = nn.Dropout(config.resid_pdrop)
 
    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)
 
 
class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        # config对应的GPT2Config()类中, n_embd属性默认为768, 因此此处hidden_size即为768.
        hidden_size = config.n_embd
        # config对应的GPT2Config()类中, n_inner属性默认为None, 因此此处inner_dim一般都为4 * hidden_size.
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
 
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # 此处n_ctx即等于config对应的GPT2Config()类中的n_ctx属性, 其值为1024.
        self.attn = Attention(hidden_size, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
 
        if config.add_cross_attention:
            self.crossattention = Attention(hidden_size, n_ctx, config, scale, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
 
        self.mlp = MLP(inner_dim, config)
 
    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        ''''''
        '''
        <1> 此时的隐藏状态hidden_states的形状为 (batch_size, 1, nx), 此时nx = n_state = n_embed = all_head_size = 768，
            即此时隐藏状态hidden_states的形状为(batch_size, 1, 768)。
        <2> 此时layer_past为一个存储着past_key张量与past_value张量的大张量, 其
             形状为(2, batch_size, num_head, sql_len, head_features).
        <3> attention_mask张量为注意力遮罩张量, 其让填充特殊符[PAD]处的注意力分数极小,
             其embedding嵌入值基本不会在多头注意力聚合操作中被获取到.
        '''
 
        # 将此时输入的隐藏状态hidden_states先输入进LayerNormalization层进行层标准化计算后,
        # 再将标准化结果输入进'多头注意力计算层self.attn()'中进行多头注意力聚合操作计算.
        # 此时返回的attn_outputs列表中:
        # <1> 第一个值为多头注意力聚合操作结果张量a, 形状为(batch_size, 1, all_head_size), all_head_size=n_state=nx=n_embd=768.
        # <2> 第二个值为上方的present张量, 其存储着past_key张量与这次迭代的key张量合并后的新key张量, 以及
        #     past_value张量与这次迭代的value张量合并后的新value张量, 其形状为(2, batch_size, num_head, sql_len+1, head_features).
        # <3> 若output_attentions为True, 则第三个值为attn_outputs列表中的注意力分数张量w.
        attn_outputs = self.attn(
            self.ln_1(hidden_states),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
 
        # 此时的attn_output张量为返回的attn_outputs列表中第一个值:
        # 多头注意力聚合操作结果张量a, 形状为(batch_size, 1, all_head_size), all_head_size=n_state=nx=n_embd=768.
        attn_output = attn_outputs[0]  # output_attn列表: a, present, (attentions)
        outputs = attn_outputs[1:]
 
        # residual connection, 进行残差连接.
        # 此时attn_output张量形状为(batch_size, 1, all_head_size), all_head_size=n_state=nx=n_embd=768.
        # hidden_states的形状为(batch_size, 1, 768).
        hidden_states = attn_output + hidden_states
 
 
        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
 
 
            '''此时self.crossattention()的Cross_Attention运算过程与self.attn()的Attention运算过程几乎相同, 其不同点在于：
 
            <1> self.attn()的Attention运算是将LayerNormalization之后的hidden_states通过'self.c_attn = Conv1D(3 * n_state, nx)
            (第173行代码)'将hidden_states的形状由(batch_size, 1, 768)投影为(batch_size, 1, 3 * 768), 再将投影后的hidden_states
            在第三维度(dim=2)上拆分为三份, 将其分别赋为query、key、value, 其形状都为(batch_size, 1, 768)；
            此时n_state = nx = num_head*head_features = 768.
             
            之后经过split_heads()函数拆分注意力头且key、value张量分别与past_key、past_value张量合并之后:
            query张量的形状变为(batch_size, num_head, 1, head_features),
            key张量的形状变为(batch_size, num_head, head_features, sql_len+1),
            value张量的形状变为(batch_size, num_head, sql_len+1, head_features).
 
            <2> self.crossattention()的Cross_Attention运算过程则是将LayerNormalization之后的hidden_states通过
            'self.q_attn = Conv1D(n_state, nx)(第168行代码)'将hidden_states的形状由(batch_size,1, 768)投影为(batch_size,1, 768),
            将此投影之后的hidden_states赋值作为query张量；
            再将此时从编码器(encoder)中传过来的编码器隐藏状态encoder_hidden_states通过'self.c_attn = Conv1D(2 * n_state, nx)
            (第164行代码)'将encoder_hidden_states的形状由(batch_size, enc_seq_len, 768)投影为(batch_size, enc_seq_len, 2 * 768),
            将投影后的encoder_hidden_states在在第三维度(dim=2)上拆分为两份分别赋为key、value,
            其形状都为(batch_size, enc_seq_len, 768)；  此时n_state = nx = num_head*head_features = 768.
             
            之后经过split_heads()函数拆分注意力头之后:
            query张量的形状变为(batch_size, num_head, 1, head_features),
            key张量的形状变为(batch_size, num_head, head_features, enc_seq_len),
            value张量的形状变为(batch_size, num_head, enc_seq_len, head_features).
            此时计算出的cross_attention张量形状为(batch_size, num_head, 1, enc_seq_len).'''
 
            # 此时将上方的隐藏状态hidden_states(Attention运算结果+Attention运算前的hidden_states)先输入进LayerNormalization
            # 层进行层标准化计算后, 再将标准化结果输入进'交叉多头注意力计算层self.crossattention()'中与编码器传入的隐藏状态
            # encoder_hidden_states进行交叉多头注意力聚合操作计算.
            # 此时返回的cross_attn_outputs列表中:
            # <1> 第一个值为与编码器传入的隐藏状态encoder_hidden_states进行交叉多头注意力聚合操作的结果张量a,
            #     形状为(batch_size, 1, all_head_size), all_head_size=n_state=nx=n_embd=768。
            # <2> 第二个值仍为present张量, 但由于此时是做'交叉多头注意力计算self.crossattention()',此时输入进self.crossattention()
            #     函数的参数中不包含layer_past(来自past_key_values列表)的past_key与past_value张量, 因此此时的present为(None,),
            #     详细代码可见本脚本代码357行, 因此此处用不到'交叉多头注意力计算结果列表cross_attn_outputs'中的present,
            #     将其舍弃(代码第528行)。
            # <3> 若output_attentions为True, 则第三个值为: 交叉注意力分数张量w, 即cross attentions,
            #      cross_attention张量形状为(batch_size, num_head, 1, enc_seq_len).
            cross_attn_outputs = self.crossattention(
                self.ln_cross_attn(hidden_states),
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = hidden_states + attn_output
            # cross_attn_outputs[2:] add cross attentions if we output attention weights,
            # 即将'交叉多头注意力计算结果列表cross_attn_outputs'中的交叉注意力分数张量cross_attention保存为此时的
            # outputs列表中的最后一个元素.
            outputs = outputs + cross_attn_outputs[2:]
 
 
        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))
        # residual connection
        hidden_states = hidden_states + feed_forward_hidden_states
 
        outputs = [hidden_states] + outputs
 
        # 此时返回的outputs列表中的元素为：
        # <1> 第一个值为多头注意力聚合操作结果张量hidden_states输入前馈MLP层与残差连接之后得到的最终hidden_states张量,
        #     形状为(batch_size, 1, n_state), all_head_size=n_state=nx=n_embd=768.
        # <2> 第二个值为上方的present张量, 其存储着past_key张量与这次迭代的key张量合并后的新key张量, 以及
        #     past_value张量与这次迭代的value张量合并后的新value张量, 其形状为(2, batch_size, num_head, sql_len+1, head_features).
        # <3> 若output_attentions为True, 则第三个值为attn_outputs列表中的注意力分数张量w.
        # <4> 若此时进行了Cross Attention计算, 则第四个值为'交叉多头注意力计算结果列表cross_attn_outputs'中的
        #     交叉注意力分数张量cross_attention, 其形状为(batch_size, num_head, 1, enc_seq_len).
        return outputs  # hidden_states, present, (attentions, cross_attentions)
 
 
class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
 
    config_class = GPT2Config
    load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = "transformer"
 
    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
 
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
 
 
@dataclass
class GPT2DoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.
 
    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided):
            Language modeling loss.
        mc_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`mc_labels` is provided):
            Multiple choice classification loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`, with each tensor of shape :obj:`(2,
            batch_size, num_heads, sequence_length, embed_size_per_head)`).
 
            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            :obj:`past_key_values` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
 
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
 
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
 
    loss: Optional[torch.FloatTensor] = None
    mc_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mc_logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
 
 
GPT2_START_DOCSTRING = r"""
 
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)
 
    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.
 
    Parameters:
        config (:class:`~transformers.GPT2Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""
 
GPT2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, input_ids_length)`):
            :obj:`input_ids_length` = ``sequence_length`` if :obj:`past_key_values` is ``None`` else
            ``past_key_values[0].shape[-2]`` (``sequence_length`` of input past key value states). Indices of input
            sequence tokens in the vocabulary.
 
            If :obj:`past_key_values` is used, only ``input_ids`` that do not have their past calculated should be
            passed as ``input_ids``.
 
            Indices can be obtained using :class:`~transformers.GPT2Tokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.
 
            `What are input IDs? <../glossary.html#input-ids>`__
        past_key_values (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            :obj:`past_key_values` output below). Can be used to speed up sequential decoding. The ``input_ids`` which
            have their past given to this model should not be passed as ``input_ids`` as they have already been
            computed.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
 
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
 
            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, input_ids_length)`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:
 
            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.
 
            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.
 
            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:
 
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
 
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
 
            If :obj:`past_key_values` is used, optionally only the last :obj:`inputs_embeds` have to be input (see
            :obj:`past_key_values`).
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""
 
 
@add_start_docstrings(
    "The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.",
    GPT2_START_DOCSTRING,
)
 
class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
 
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
 
        self.init_weights()
 
    def get_input_embeddings(self):
        return self.wte
 
    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings
 
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)
 
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="gpt2",
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
 
        # input_ids与inputs_embeds只能输入一个，有input_ids变只需将input_ids输入嵌入层即可变为类似inputs_embeds的张量,
        # 有inputs_embeds变不需要input_ids
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
 
        # 下方是确保输入的input_ids、token_type_ids、position_ids等张量的形状为正确的样式:
        # <1> 若为模型第一次迭代, 则此时input_ids、token_type_ids、position_ids等张量的正确形状为 (batch_size, seq_len),
        # <2> 若为模型第二次及之后的迭代, 则此时input_ids、token_type_ids、position_ids等张量的正确形状为 (batch_size, 1).
        # 最后, 将输入的input_ids、token_type_ids、position_ids等张量的形状保存到input_shape中.
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
 
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])
 
        if past_key_values is None:
            past_length = 0
            # 若此时为GPT2模型第一次迭代, 则不存在上一次迭代返回的past_key_values列表(包含12个present的列表,
            # 也就是代码中的presents列表), 则此时past_key_values列表为一个包含12个None值的列表.
            past_key_values = [None] * len(self.h)
        else:
            past_length = past_key_values[0][0].size(-2)
 
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            '''<1> GPT2Model第一次迭代时输入GPT2Model的forward()函数中的past_key_values参数为None, 此时past_length为0, 
              input_shape[-1] + past_length就等于第一次迭代时输入的文本编码(input_ids)的seq_len维度本身, 
              此时创建的position_ids张量形状为(batch_size, seq_len).
              <2> 若为GPT2Mode第二次及之后的迭代时, 此时past_length为上一次迭代时记录保存下来的past_key_values中
              张量的seq_len维度, 而input_shape[-1] + past_length则等于seq_len + 1, 因为在第二次及之后的迭代中,
              输入的文本编码(input_ids)的seq_len维度本身为1,即第二次及之后的迭代中每次只输入一个字的文本编码,
              此时创建的position_ids张量形状为(batch_size, 1).'''
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
 
        # Attention mask.
        # attention_mask张量为注意力遮罩张量, 其让填充特殊符[PAD]处的注意力分数极小,其embedding嵌入值
        # 基本不会在多头注意力聚合操作中被获取到.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]
 
            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0
 
        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length],
        # 若此时有从编码器encoder中传入的编码器隐藏状态encoder_hidden_states, 则获取编码器隐藏状态encoder_hidden_states
        # 的形状(encoder_batch_size, encoder_sequence_length), 同时定义编码器隐藏状态对应的attention_mask张量(即encoder_attention_mask).
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            # If a 2D ou 3D attention mask is provided for the cross-attention
            # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None
 
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        # prune_heads()可结合 https://github.com/huggingface/transformers/issues/850 理解.
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
 
        # 将input_ids、token_type_ids、position_ids等张量输入嵌入层self.wte()、 self.wpe()中之后获取其嵌入形式张量
        # inputs_embeds、position_embeds与token_type_embeds.
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
 
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds
 
        '''<1> GPT2Model第一次迭代时输入GPT2Model的forward()函数中的past_key_values参数为None, 此时past_length为0, 
              此时hidden_states张量形状为(batch_size, sel_len, n_embd)，config的GPT2Config()类中n_emb默认为768.
          <2> 若为GPT2Mode第二次及之后的迭代时, 此时past_length为上一次迭代时记录保存下来的past_key_values中
              张量的seq_len维度, 而input_shape[-1] + past_length则等于seq_len + 1, 因为在第二次及之后的迭代中,
              输入的文本编码(input_ids)的seq_len维度本身为1,即第二次及之后的迭代中每次只输入一个字的文本编码,
              此时hidden_states张量形状为(batch_size, 1, n_embd)，config的GPT2Config()类中n_emb默认为768.'''
        hidden_states = self.drop(hidden_states)
 
        output_shape = input_shape + (hidden_states.size(-1),)
 
        # config对应的GPT2Config()类中的use_cache默认为True.
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
 
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            '''此处past_key_values元组中一共有12个元素(layer_past), 分别对应GPT2模型中的12层Transformer_Block,
            每一个layer_past都为模型上一次迭代中每个Transformer_Block保留下来的present张量, 而每个present张量保存着
            Transformer_Block中Attention模块将本次迭代的key张量与上一次迭代中的past_key张量(layer_past[0])合并、
            将本次迭代的value张量与上一次迭代中的past_value张量(layer_past[1])合并所得的新的key张量与value张量,
            之后保存着本次迭代中12层Transformer_Block每一层中返回的present张量的presents元组, 便会被作为下一次迭代中
            的past_key_values元组输入进下一次迭代的GPT2模型中。
            新的key张量与value张量详细解析如下：'''
 
            '''第一次迭代时query、key、value张量的seq_len维度处的维度数就为seq_len而不是1, 第二次之后seq_len维度的维度数皆为1.'''
 
            '''<1> 本次迭代中新的key张量
            此时需要通过layer_past[0].transpose(-2, -1)操作将past_key张量的形状变为(batch_size, num_head, head_features, sql_len),
            而此时key张量的形状为(batch_size, num_head, head_features, 1), 这样在下方就方便将past_key张量与key张量在最后
            一个维度(dim=-1)处进行合并, 这样就将当前token的key部分加入了past_key的seq_len部分, 以方便模型在后面预测新的token,
            此时新的key张量的形状为: (batch_size, num_head, head_features, sql_len+1), new_seq_len为sql_len+1。
             <2>  本次迭代中新的value张量
            而此时past_value(layer_past[1])不用变形, 其形状为(batch_size, num_head, sql_len, head_features), 
            而此时value张量的形状为(batch_size, num_head, 1, head_features), 这样在下方就方便将past_value张量与value张量
            在倒数第二个维度(dim=-2)处进行合并, 这样就将当前token的value部分加入了past_value的seq_len部分, 
            以方便模型在后面预测新的token,
            此时新的value张量的形状为: (batch_size, num_head, sql_len+1, head_features), new_seq_len为sql_len+1。'''
 
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)
 
            if getattr(self.config, "gradient_checkpointing", False):
 
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # checkpointing only works with tuple returns, not with lists
                        return tuple(output for output in module(*inputs, use_cache, output_attentions))
 
                    return custom_forward
 
                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    layer_past,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                # 此时返回的outputs列表中的元素为：
                # <1> 第一个值为多头注意力聚合操作结果张量hidden_states输入前馈MLP层与残差连接之后得到的hidden_states张量,
                #     形状为(batch_size, 1, n_state), all_head_size=n_state=nx=n_embd=768.
                # <2> 第二个值为上方的present张量, 其存储着past_key张量与这次迭代的key张量合并后的新key张量, 以及
                #     past_value张量与这次迭代的value张量合并后的新value张量, 其形状为(2, batch_size, num_head, sql_len+1, head_features).
                # <3> 若output_attentions为True, 则第三个值为attn_outputs列表中的注意力分数张量w.
                # <4> 若此时进行了Cross Attention计算, 则第四个值为'交叉多头注意力计算结果列表cross_attn_outputs'中的
                #     交叉注意力分数张量cross_attention, 其形状为(batch_size, num_head, 1, enc_seq_len).
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
 
            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)
 
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3],)
 
        # 将PT2模型中12层Block模块计算后得到的最终hidden_states张量再输入进LayerNormalization层中进行计算.
        hidden_states = self.ln_f(hidden_states)
 
        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state, 即将上方最后一层Block()循环结束之后得到的结果隐藏状态张量hidden_states
        # 也添加入元组all_hidden_states中.
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
 
        # 此时返回的元素为：
        # <1> 第一个值为GPT2模型中经过12层Block模块计算后得到的最终hidden_states张量,
        #     形状为(batch_size, 1, n_state), all_head_size=n_state=nx=n_embd=768.
        # <2> 第二个值为GPT2模型中12层Block模块计算后得到的存储12个present张量的presents元组, 每一个present张量存储着
        #     past_key张量与这次迭代的key张量合并后的新key张量, 以及past_value张量与这次迭代的value张量合并后的新value张量,
        #     一个present张量形状为(2, batch_size, num_head, sql_len+1, head_features).
        # <3> 若output_hidden_states为True, 则第三个值为GPT2模型中12层Block模块计算后得到的存储12个隐藏状态张量hidden_states
        #     的all_hidden_states元组.
        # <4> 若output_attentions为True, 则第四个值为GPT2模型中12层Block模块计算后得到的存储12个注意力分数张量w
        #     的all_self_attentions元组.
        # <5> 若此时进行了Cross Attention计算, 则第五个值为GPT2模型中12层Block模块计算后得到的存储12个交叉注意力分数张量
        #     cross_attention的all_cross_attentions元组,
        #     其中每个交叉注意力分数张量cross_attention形状为(batch_size, num_head, 1, enc_seq_len).
        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)
 
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
 
 
@add_start_docstrings(
    """
    The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPT2_START_DOCSTRING,
)
 
 
class GPT2LMHeadModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]
 
    def __init__(self, config):
        super().__init__(config)
 
        # 初始化GPT2Model(config)类.
        self.transformer = GPT2Model(config)
 
        # self.lm_head为将GPT2Model(config)计算输出的hidden_states张量的最后一个维度由768维(config.n_embd)投影为
        # 词典大小维度(config.vocab_size)的输出层, 此时hidden_states张量的形状将会由(batch_size, 1, n_embed)投影变为
        # lm_logits张量的(batch_size, 1, vocab_size).
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
 
        # 重新初始化权重矩阵.
        self.init_weights()
 
    def get_output_embeddings(self):
        return self.lm_head
 
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
 
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
 
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
 
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="gpt2",
        output_type=CausalLMOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
 
        # 此时返回的transformer_outputs中为：
        # <1> 第一个值为GPT2模型中经过12层Block模块计算后得到的最终hidden_states张量,
        #     形状为(batch_size, 1, n_state), all_head_size=n_state=nx=n_embd=768.
        # <2> 第二个值为GPT2模型中12层Block模块计算后得到的存储12个present张量的presents元组, 每一个present张量存储着
        #     past_key张量与这次迭代的key张量合并后的新key张量, 以及past_value张量与这次迭代的value张量合并后的新value张量,
        #     一个present张量形状为(2, batch_size, num_head, sql_len+1, head_features).
        # <3> 若output_hidden_states为True, 则第三个值为GPT2模型中12层Block模块计算后得到的存储12个隐藏状态张量hidden_states
        #     的all_hidden_states元组.
        # <4> 若output_attentions为True, 则第四个值为GPT2模型中12层Block模块计算后得到的存储12个注意力分数张量w
        #     的all_self_attentions元组.
        # <5> 若此时进行了Cross Attention计算, 则第五个值为GPT2模型中12层Block模块计算后得到的存储12个交叉注意力分数张量
        #     cross_attention的all_cross_attentions元组,
        #     其中每个交叉注意力分数张量cross_attention形状为(batch_size, num_head, 1, enc_seq_len).
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
 
        # self.lm_head()输出层将GPT2Model(config)计算输出的hidden_states张量的最后一个维度由768维(config.n_embd)
        # 投影为词典大小维度(config.vocab_size)的输出层, 此时hidden_states张量的形状将会由(batch_size, 1, n_embed)投影变为
        # lm_logits张量的(batch_size, 1, vocab_size).
        lm_logits = self.lm_head(hidden_states)
 
        loss = None
        # 若此时labels也输入进了GPT2LMHeadModel()类中, 则此时会使用自回归的方式计算交叉熵损失,
        # 即此时的shift_logits为将GPT2Model(config)计算输出的hidden_states张量的最后一个维度由768维(config.n_embd)投影为
        # 词典大小维度(config.vocab_size)所得到的lm_logits张量的切片lm_logits[..., :-1, :].contiguous(),即取(1, n-1)的lm_logits值；
        # 此时的shift_labels为将输入的labels张量的切片labels[..., 1:].contiguous(), 即取(2, n)的label值；
        # 因此利用(1, n-1)的lm_logits值与(2, n)的label值即可计算此时自回归预训练的交叉熵损失值.
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
 
 
        # <1> 若loss不为None, 则代表此时输入了labels张量, 进行了自回归的交叉熵损失计算, 则此时第一个值为
        #     自回归交叉熵损失loss.
        # <2> 第二个值将GPT2Model(config)计算输出的hidden_states张量的最后一个维度由768维(config.n_embd)投影为
        #     词典大小维度(config.vocab_size)的lm_logits张量, 其形状为(batch_size, 1, vocab_size).
        # <3> 第三个值为GPT2模型中12层Block模块计算后得到的存储12个present张量的presents元组, 每一个present张量存储着
        #     past_key张量与这次迭代的key张量合并后的新key张量, 以及past_value张量与这次迭代的value张量合并后的新value张量,
        #     一个present张量形状为(2, batch_size, num_head, sql_len+1, head_features).
        # <4> 若output_hidden_states为True, 则第四个值为GPT2模型中12层Block模块计算后得到的存储12个隐藏状态张量hidden_states
        #     的all_hidden_states元组.
        # <5> 若output_attentions为True, 则第五个值为GPT2模型中12层Block模块计算后得到的存储12个注意力分数张量w
        #     的all_self_attentions元组.
        # <6> 若此时进行了Cross Attention计算, 则第六个值为GPT2模型中12层Block模块计算后得到的存储12个交叉注意力分数张量
        #     cross_attention的all_cross_attentions元组,
        #     其中每个交叉注意力分数张量cross_attention形状为(batch_size, num_head, 1, enc_seq_len).
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
 
        return CausalLMOutputWithPastAndCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
 
 
@add_start_docstrings(
    """
The GPT2 Model transformer with a language modeling and a multiple-choice classification head on top e.g. for
RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the
input embeddings, the classification head takes as input the input of a specified classification token index in the
input sequence).
""",
    GPT2_START_DOCSTRING,
)
class GPT2DoubleHeadsModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.multiple_choice_head = SequenceSummary(config)
 
        self.init_weights()
 
    def get_output_embeddings(self):
        return self.lm_head
 
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
 
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
 
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
 
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
 
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=GPT2DoubleHeadsModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        labels=None,
        mc_labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        mc_token_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_choices)`, `optional`, default to index of the last token of the input):
            Index of the classification token in each input sequence. Selected in the range ``[0, input_ids.size(-1) -
            1[``.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-1, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        mc_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices]`` where `num_choices` is the size of the second dimension of the input tensors. (see
            `input_ids` above)
 
        Return:
 
        Example::
 
            >>> import torch
            >>> from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel
 
            >>> tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            >>> model = GPT2DoubleHeadsModel.from_pretrained('gpt2')
 
            >>> # Add a [CLS] to the vocabulary (we should train it also!)
            >>> num_added_tokens = tokenizer.add_special_tokens({'cls_token': '[CLS]'})
 
            >>> embedding_layer = model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size
 
            >>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
            >>> encoded_choices = [tokenizer.encode(s) for s in choices]
            >>> cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]
 
            >>> input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
            >>> mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1
 
            >>> outputs = model(input_ids, mc_token_ids=mc_token_ids)
            >>> lm_logits = outputs.lm_logits
            >>> mc_logits = outputs.mc_logits
 
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
 
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
 
        hidden_states = transformer_outputs[0]
 
        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)
 
        mc_loss = None
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            mc_loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
        lm_loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
 
        if not return_dict:
            output = (lm_logits, mc_logits) + transformer_outputs[1:]
            if mc_loss is not None:
                output = (mc_loss,) + output
            return ((lm_loss,) + output) if lm_loss is not None else output
 
        return GPT2DoubleHeadsModelOutput(
            loss=lm_loss,
            mc_loss=mc_loss,
            logits=lm_logits,
            mc_logits=mc_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
 
 
@add_start_docstrings(
    """
    The GPT2 Model transformer with a sequence classification head on top (linear layer).
 
    :class:`~transformers.GPT2ForSequenceClassification` uses the last token in order to do the classification, as
    other causal models (e.g. GPT-1) do.
 
    Since it does classification on the last token, it requires to know the position of the last token. If a
    :obj:`pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each
    row. If no :obj:`pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot
    guess the padding tokens when :obj:`inputs_embeds` are passed instead of :obj:`input_ids`, it does the same (take
    the last value in each row of the batch).
    """,
    GPT2_START_DOCSTRING,
)
class GPT2ForSequenceClassification(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]
 
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)
 
        self.init_weights()
 
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="microsoft/dialogrpt",
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
 
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)
 
        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]
 
        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
 
 
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    f"unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )
 
        pooled_logits = logits[range(batch_size), sequence_lengths]
 
        loss = None
        if labels is not None:
            # 若此时每条样本数据只有一个标签, 则认为此时是做回归任务.
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(pooled_logits.view(-1), labels.to(self.dtype).view(-1))
 
            # 若此时每条样本数据大于一个标签, 则认为此时是做分类任务.
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
 
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
 
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )