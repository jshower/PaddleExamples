import sys
import os
import math

import paddle.fluid as fluid
from paddle.fluid.initializer import NormalInitializer


def ner_net(word_dict_len, label_dict_len, parallel):
    IS_SPARSE = False
    #embedding_name = 'emb'
    #word_dict_len = 1942562
    word_dim = 32
    mention_dict_len = 57
    mention_dim = 20
    grnn_hidden = 36
    #label_dict_len = 49
    emb_lr = 5
    init_bound = 0.1 
    def _net_conf(word, mark, target):
        word_embedding = fluid.layers.embedding(
            input=word,
            size=[word_dict_len, word_dim],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr=fluid.ParamAttr(learning_rate=emb_lr, name="word_emb", initializer=fluid.initializer.Uniform(low=-init_bound, high=init_bound)
        #        gradient_clip=fluid.clip.GradientClipByValue(max=0.01, min=-0.05)
        ))

        mention_embedding = fluid.layers.embedding(
            input=mention,
            size=[mention_dict_len, mention_dim],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr=fluid.ParamAttr(learning_rate=emb_lr, name="mention_emb", initializer=fluid.initializer.Uniform(low=-init_bound, high=init_bound)
        #        gradient_clip=fluid.clip.GradientClipByValue(max=0.01, min=-0.05)
        ))

        word_embedding_r = fluid.layers.embedding(
            input=word,
            size=[word_dict_len, word_dim],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr=fluid.ParamAttr(learning_rate=emb_lr, name="word_emb_r", initializer=fluid.initializer.Uniform(low=-init_bound, high=init_bound)
        #        gradient_clip=fluid.clip.GradientClipByValue(max=0.01, min=-0.05)
        ))

        mention_embedding_r = fluid.layers.embedding(
            input=mention,
            size=[mention_dict_len, mention_dim],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr=fluid.ParamAttr(learning_rate=emb_lr, name="mention_emb_r", initializer=fluid.initializer.Uniform(low=-init_bound, high=init_bound)
        #        gradient_clip=fluid.clip.GradientClipByValue(max=0.01, min=-0.05)
        ))

        word_mention_vector = fluid.layers.concat(
            input=[word_embedding, mention_embedding], axis=1)

        word_mention_vector_r = fluid.layers.concat(
            input=[word_embedding_r, mention_embedding_r], axis=1)
    
        pre_gru = fluid.layers.fc(input = word_mention_vector, 
            size = grnn_hidden * 3,
            param_attr = fluid.ParamAttr(initializer=fluid.initializer.Uniform(low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(regularization_coeff=1e-4)))
        gru = fluid.layers.dynamic_gru(input=pre_gru, 
            size=grnn_hidden,
            param_attr = fluid.ParamAttr(initializer=fluid.initializer.Uniform(low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(regularization_coeff=1e-4)))

        pre_gru_r = fluid.layers.fc(input=word_mention_vector_r, 
            size=grnn_hidden * 3,
            param_attr = fluid.ParamAttr(initializer=fluid.initializer.Uniform(low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(regularization_coeff=1e-4)))
        gru_r = fluid.layers.dynamic_gru(input=pre_gru_r,
            size=grnn_hidden,
            is_reverse=True,
            param_attr = fluid.ParamAttr(initializer=fluid.initializer.Uniform(low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(regularization_coeff=1e-4)))

        gru_merged = fluid.layers.concat(input=[gru, gru_r], axis=1)

        emission = fluid.layers.fc(
            size=label_dict_len,
            input=gru_merged,
            param_attr = fluid.ParamAttr(initializer=fluid.initializer.Uniform(low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(regularization_coeff=1e-4)))

        crf_cost = fluid.layers.linear_chain_crf(
            input=emission,
            label=target,
            param_attr=fluid.ParamAttr(
                name='crfw',
                learning_rate=0.2,
                #regularizer=fluid.regularizer.L2DecayRegularizer(regularization_coeff=1e-4)
        ))
        avg_cost = fluid.layers.mean(x=crf_cost)
        #fluid.clip.set_gradient_clip(clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0),  param_list=["word_emb", "mention_emb", "word_emb_r", "mention_emb_r"])
        return avg_cost, emission

    word = fluid.layers.data(name='word', shape=[1], dtype='int64', lod_level=1)
    mention = fluid.layers.data(name='mention', shape=[1], dtype='int64', lod_level=1)
    target = fluid.layers.data(
        name="target", shape=[1], dtype='int64', lod_level=1)

    avg_cost, emission = _net_conf(word, mention, target)

    return avg_cost, emission, word, mention, target
