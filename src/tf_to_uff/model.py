# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import tensorflow as tf
import numpy as np

BATCH_SIZE = 1
SEQ_LEN = 32
ENC_LEN = 128
EMB_LEN = 200
DIM = 128
VOC_LEN = 4096


# Q -> embedding -> LSTMEncoder -> MapEncoder -> LSTMDecoder -> MapDecoder -> A
def Embedding(params):
    W = tf.get_variable(name='Embedding.W',initializer=tf.random_normal([VOC_LEN, EMB_LEN]))
    params.extend([W])
    def unit(x):
        return tf.matmul(x, W)
    return unit


def LSTMEncoder(params):
    Wi  = tf.get_variable(name='Encoder.Wi',initializer=tf.random_normal([EMB_LEN, DIM]))
    Ui  = tf.get_variable(name='Encoder.Ui',initializer=tf.random_normal([DIM, DIM]))
    bi  = tf.get_variable(name='Encoder.bi',initializer=tf.random_normal([DIM]))

    Wf  = tf.get_variable(name='Encoder.Wf',initializer=tf.random_normal([EMB_LEN, DIM]))
    Uf  = tf.get_variable(name='Encoder.Uf',initializer=tf.random_normal([DIM, DIM]))
    bf  = tf.get_variable(name='Encoder.bf',initializer=tf.random_normal([DIM]))

    Wog = tf.get_variable(name='Encoder.Wog',initializer=tf.random_normal([EMB_LEN, DIM]))
    Uog = tf.get_variable(name='Encoder.Uog',initializer=tf.random_normal([DIM, DIM]))
    bog = tf.get_variable(name='Encoder.bog',initializer=tf.random_normal([DIM]))

    Wc  = tf.get_variable(name='Encoder.Wc',initializer=tf.random_normal([EMB_LEN, DIM]))
    Uc  = tf.get_variable(name='Encoder.Uc',initializer=tf.random_normal([DIM, DIM]))
    bc  = tf.get_variable(name='Encoder.bc',initializer=tf.random_normal([DIM]))
    params.extend([Wi,  Ui,  bi, Wf,  Uf,  bf, Wog, Uog, bog, Wc,  Uc,  bc])

    def unit(x, previous_hidden_state, c_prev):
        c_prev = tf.matmul(c_prev, identity)
        k = tf.sigmoid(
            tf.matmul(x, Wi) +
            tf.matmul(previous_hidden_state, Ui) + bi
        )

        i = tf.sigmoid(
            tf.matmul(x, Wi) +
            tf.matmul(previous_hidden_state, Ui) + bi
        )
        f = tf.sigmoid(
            tf.matmul(x, Wf) +
            tf.matmul(previous_hidden_state, Uf) + bf
        )
        o = tf.sigmoid(
            tf.matmul(x, Wog) +
            tf.matmul(previous_hidden_state, Uog) + bog
        )
        c_ = tf.nn.tanh(
             tf.matmul(x, Wc) +
             tf.matmul(previous_hidden_state, Uc) + bc
        )
        c = tf.add(f*c_prev, i*c_, name='c0_out')
        current_hidden_state = tf.multiply(o, tf.nn.tanh(c), name='h0_out')
        return current_hidden_state
    return unit


def MapEncoder(params):
    Wo = tf.get_variable(name='Encoder.Wo',initializer=tf.random_normal([DIM, ENC_LEN]))
    bo = tf.get_variable(name='Encoder.bo',initializer=tf.random_normal([ENC_LEN]))
    params.extend([Wo, bo])

    def unit(hidden_state):
        logits = tf.matmul(hidden_state, Wo) + bo
        return logits
    return unit


def LSTMDecoder(params):
    Wi  = tf.get_variable(name='Generator.Wi',initializer=tf.random_normal([EMB_LEN+ENC_LEN, DIM]))
    Ui  = tf.get_variable(name='Generator.Ui',initializer=tf.random_normal([DIM, DIM]))
    bi  = tf.get_variable(name='Generator.bi',initializer=tf.random_normal([DIM]))

    Wf  = tf.get_variable(name='Generator.Wf',initializer=tf.random_normal([EMB_LEN+ENC_LEN, DIM]))
    Uf  = tf.get_variable(name='Generator.Uf',initializer=tf.random_normal([DIM, DIM]))
    bf  = tf.get_variable(name='Generator.bf',initializer=tf.random_normal([DIM]))

    Wog = tf.get_variable(name='Generator.Wog',initializer=tf.random_normal([EMB_LEN+ENC_LEN, DIM]))
    Uog = tf.get_variable(name='Generator.Uog',initializer=tf.random_normal([DIM, DIM]))
    bog = tf.get_variable(name='Generator.bog',initializer=tf.random_normal([DIM]))

    Wc  = tf.get_variable(name='Generator.Wc',initializer=tf.random_normal([EMB_LEN+ENC_LEN, DIM]))
    Uc  = tf.get_variable(name='Generator.Uc',initializer=tf.random_normal([DIM, DIM]))
    bc  = tf.get_variable(name='Generator.bc',initializer=tf.random_normal([DIM]))
    params.extend([Wi,  Ui,  bi, Wf,  Uf,  bf, Wog, Uog, bog, Wc,  Uc,  bc])

    def unit(x, previous_hidden_state, c_prev, code):
        c_prev = tf.matmul(c_prev, identity)
        x = tf.concat([x,code], 1)
        i = tf.sigmoid(
            tf.matmul(x, Wi) +
            tf.matmul(previous_hidden_state, Ui) + bi
        )
        f = tf.sigmoid(
            tf.matmul(x, Wf) +
            tf.matmul(previous_hidden_state, Uf) + bf
        )
        o = tf.sigmoid(
            tf.matmul(x, Wog) +
            tf.matmul(previous_hidden_state, Uog) + bog
        )
        c_ = tf.nn.tanh(
            tf.matmul(x, Wc) +
            tf.matmul(previous_hidden_state, Uc) + bc
        )
        c = tf.add(f*c_prev, i*c_, name='c1_out')
        current_hidden_state = tf.multiply(o, tf.nn.tanh(c), name='h1_out')
        return current_hidden_state
    return unit


def MapDecoder(params):
    Wo = tf.get_variable(name='Generator.Wo',initializer=tf.random_normal([DIM, VOC_LEN]))
    bo = tf.get_variable(name='Generator.bo',initializer=tf.random_normal([VOC_LEN]))
    params.extend([Wo, bo])

    def unit(hidden_state):
        logits = tf.matmul(hidden_state, Wo) + bo
        return logits
    return unit


def getChatBotModel(filepath):
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        tf.train.Saver().restore(session, filepath)
        graphdef = tf.get_default_graph().as_graph_def()
        frozen_graph = tf.graph_util.convert_variables_to_constants(session, graphdef, ['h0_out','c0_out','h1_out','c1_out','final_output'])
        return tf.graph_util.remove_training_nodes(frozen_graph)


enc_text = tf.placeholder(tf.float32, shape=[BATCH_SIZE, VOC_LEN], name='enc_text')
dec_text = tf.placeholder(tf.float32, shape=[BATCH_SIZE, VOC_LEN], name='dec_text')
h0 = tf.placeholder(tf.float32, shape=[BATCH_SIZE, DIM], name='h0_in')
c0 = tf.placeholder(tf.float32, shape=[BATCH_SIZE, DIM], name='c0_in')
h1 = tf.placeholder(tf.float32, shape=[BATCH_SIZE, DIM], name='h1_in')
c1 = tf.placeholder(tf.float32, shape=[BATCH_SIZE, DIM], name='c1_in')

gen_params = []
embed_unit = Embedding(gen_params)
enc_rec_unit = LSTMEncoder(gen_params)
enc_map_unit = MapEncoder(gen_params)
dec_rec_unit = LSTMDecoder(gen_params)
dec_map_unit = MapDecoder(gen_params)

identity = tf.constant(np.identity(DIM, np.float32))
enc_embed = embed_unit(enc_text)
dec_embed = embed_unit(dec_text)

h0 = enc_rec_unit(enc_embed, h0, c0)
ct = enc_map_unit(h0)
h1 = dec_rec_unit(dec_embed, h1, c1, ct)
ot = dec_map_unit(h1)
prob = tf.nn.softmax(ot, name='final_output')
