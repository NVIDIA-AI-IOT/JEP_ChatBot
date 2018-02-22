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

import os, sys
sys.path.append(os.getcwd())

from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops

import time
import pickle
import numpy as np
import tensorflow as tf
import parser as Parser

BATCH_SIZE = 256
ITERS = 1000000
SEQ_LEN = 32
ENC_LEN = 128
EMB_LEN = 200
DIM = 128
CRITIC_ITERS = 10
LAMBDA = 2
DATA_NAME = 'cornell_pair'
VOC_LEN = 4096
MAX_N_EXAMPLES = 217377


if not os.path.exists("%s.pickle" % DATA_NAME):
    lines, wmap, iwmap = Parser.readCornellPair(dataset=DATA_NAME, maxline=MAX_N_EXAMPLES, maxlen=SEQ_LEN, maxvoc=VOC_LEN)
    with open("%s.pickle" % DATA_NAME, 'w') as f:
        pickle.dump([wmap, iwmap], f)
else:
    with open("%s.pickle" % DATA_NAME) as f:
        wmap, iwmap = pickle.load(f)
    lines = Parser.loadCornellPair(dataset=DATA_NAME, _wordmap=wmap, _inv_wordmap=iwmap, maxline=MAX_N_EXAMPLES, maxlen=SEQ_LEN, maxvoc=VOC_LEN)  

log = open(time.strftime("log/main_%Y%m%d_%H%M%S"),"w")


def myprint(line):
    print line
    log.write(line+"\n")


def init_matrix(shape):
    return tf.random_normal(shape, stddev=0.1)


def Embedding(params):
    W = tf.get_variable(name='Embedding.W',initializer=init_matrix([VOC_LEN, EMB_LEN]))
    params.extend([W])
    def unit(x):
        return tf.nn.embedding_lookup(W, x)
    return unit


def LSTMEncoder(params):
    Wi  = tf.get_variable(name='Encoder.Wi',initializer=init_matrix([EMB_LEN, DIM]))
    Ui  = tf.get_variable(name='Encoder.Ui',initializer=init_matrix([DIM, DIM]))
    bi  = tf.get_variable(name='Encoder.bi',initializer=init_matrix([DIM]))

    Wf  = tf.get_variable(name='Encoder.Wf',initializer=init_matrix([EMB_LEN, DIM]))
    Uf  = tf.get_variable(name='Encoder.Uf',initializer=init_matrix([DIM, DIM]))
    bf  = tf.get_variable(name='Encoder.bf',initializer=init_matrix([DIM]))

    Wog = tf.get_variable(name='Encoder.Wog',initializer=init_matrix([EMB_LEN, DIM]))
    Uog = tf.get_variable(name='Encoder.Uog',initializer=init_matrix([DIM, DIM]))
    bog = tf.get_variable(name='Encoder.bog',initializer=init_matrix([DIM]))

    Wc  = tf.get_variable(name='Encoder.Wc',initializer=init_matrix([EMB_LEN, DIM]))
    Uc  = tf.get_variable(name='Encoder.Uc',initializer=init_matrix([DIM, DIM]))
    bc  = tf.get_variable(name='Encoder.bc',initializer=init_matrix([DIM]))
    params.extend([Wi,  Ui,  bi, Wf,  Uf,  bf, Wog, Uog, bog, Wc,  Uc,  bc])

    def unit(x, hidden_memory_tm1):
        previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
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
        c = f * c_prev + i * c_
        current_hidden_state = o * tf.nn.tanh(c)
        return tf.stack([current_hidden_state, c])
    return unit


def MapEncoder(params):
    Wo = tf.get_variable(name='Encoder.Wo',initializer=init_matrix([DIM, ENC_LEN]))
    bo = tf.get_variable(name='Encoder.bo',initializer=init_matrix([ENC_LEN]))
    params.extend([Wo, bo])

    def unit(hidden_memory_tuple):
        hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
        logits = tf.matmul(hidden_state, Wo) + bo
        return logits
    return unit


def LSTMDecoder(params):
    Wi  = tf.get_variable(name='Generator.Wi',initializer=init_matrix([EMB_LEN+ENC_LEN, DIM]))
    Ui  = tf.get_variable(name='Generator.Ui',initializer=init_matrix([DIM, DIM]))
    bi  = tf.get_variable(name='Generator.bi',initializer=init_matrix([DIM]))

    Wf  = tf.get_variable(name='Generator.Wf',initializer=init_matrix([EMB_LEN+ENC_LEN, DIM]))
    Uf  = tf.get_variable(name='Generator.Uf',initializer=init_matrix([DIM, DIM]))
    bf  = tf.get_variable(name='Generator.bf',initializer=init_matrix([DIM]))

    Wog = tf.get_variable(name='Generator.Wog',initializer=init_matrix([EMB_LEN+ENC_LEN, DIM]))
    Uog = tf.get_variable(name='Generator.Uog',initializer=init_matrix([DIM, DIM]))
    bog = tf.get_variable(name='Generator.bog',initializer=init_matrix([DIM]))

    Wc  = tf.get_variable(name='Generator.Wc',initializer=init_matrix([EMB_LEN+ENC_LEN, DIM]))
    Uc  = tf.get_variable(name='Generator.Uc',initializer=init_matrix([DIM, DIM]))
    bc  = tf.get_variable(name='Generator.bc',initializer=init_matrix([DIM]))
    params.extend([Wi,  Ui,  bi, Wf,  Uf,  bf, Wog, Uog, bog, Wc,  Uc,  bc])

    def unit(x, hidden_memory_tm1, code):
        previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
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
        c = f * c_prev + i * c_
        current_hidden_state = o * tf.nn.tanh(c)
        return tf.stack([current_hidden_state, c])
    return unit


def MapDecoder(params):
    Wo = tf.get_variable(name='Generator.Wo',initializer=init_matrix([DIM, VOC_LEN]))
    bo = tf.get_variable(name='Generator.bo',initializer=init_matrix([VOC_LEN]))
    params.extend([Wo, bo])

    def unit(hidden_memory_tuple):
        hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
        logits = tf.matmul(hidden_state, Wo) + bo
        return logits
    return unit


def LSTMDiscriminator(params):
    Wi  = tf.get_variable(name='Discriminator.Wi',initializer=init_matrix([VOC_LEN*2, DIM]))
    Ui  = tf.get_variable(name='Discriminator.Ui',initializer=init_matrix([DIM, DIM]))
    bi  = tf.get_variable(name='Discriminator.bi',initializer=init_matrix([DIM]))

    Wf  = tf.get_variable(name='Discriminator.Wf',initializer=init_matrix([VOC_LEN*2, DIM]))
    Uf  = tf.get_variable(name='Discriminator.Uf',initializer=init_matrix([DIM, DIM]))
    bf  = tf.get_variable(name='Discriminator.bf',initializer=init_matrix([DIM]))

    Wog = tf.get_variable(name='Discriminator.Wog',initializer=init_matrix([VOC_LEN*2, DIM]))
    Uog = tf.get_variable(name='Discriminator.Uog',initializer=init_matrix([DIM, DIM]))
    bog = tf.get_variable(name='Discriminator.bog',initializer=init_matrix([DIM]))

    Wc  = tf.get_variable(name='Discriminator.Wc',initializer=init_matrix([VOC_LEN*2, DIM]))
    Uc  = tf.get_variable(name='Discriminator.Uc',initializer=init_matrix([DIM, DIM]))
    bc  = tf.get_variable(name='Discriminator.bc',initializer=init_matrix([DIM]))
    params.extend([Wi,  Ui,  bi, Wf,  Uf,  bf, Wog, Uog, bog, Wc,  Uc,  bc])

    def unit(x, hidden_memory_tm1):
        previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)
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
        c = f * c_prev + i * c_
        current_hidden_state = o * tf.nn.tanh(c)
        return tf.stack([current_hidden_state, c]) 
    return unit


def MapDiscriminator(params):
    Wo = tf.get_variable(name='Discriminator.Wo',initializer=init_matrix([DIM, 1]))
    bo = tf.get_variable(name='Discriminator.bo',initializer=init_matrix([1]))
    params.extend([Wo, bo])

    def unit(hidden_memory_tuple):
        hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
        logits = tf.matmul(hidden_state, Wo) + bo
        return logits
    return unit


def Sample(inputs_c, inputs_x):
    inputs = tf.transpose(inputs_c, perm=[1,0,2])
    input_token = tensor_array_ops.TensorArray(dtype=tf.float32,size=SEQ_LEN)
    input_token = input_token.unstack(inputs)

    predictions = tensor_array_ops.TensorArray(dtype=tf.float32, size=SEQ_LEN, dynamic_size=False, infer_shape=True)
    samples = tensor_array_ops.TensorArray(dtype=tf.int32, size=SEQ_LEN, dynamic_size=False, infer_shape=True)

    output = tf.transpose(inputs_x, perm=[1,0,2])
    output_token = tensor_array_ops.TensorArray(dtype=tf.float32,size=SEQ_LEN)
    output_token = output_token.unstack(output)

    def _encoder(i, h_tm1, n_tm1):
        #stage1
        x_t = input_token.read(i)
        h_t = enc_rec_unit(x_t, h_tm1)
        c_t = enc_map_unit(h_t)
        #stage2
        n_t = dec_rec_unit(pad_voc, n_tm1, c_t)
        return i+1, h_t, n_t
    _, h_t, n_t = control_flow_ops.while_loop(cond=lambda i, _1, _2: i < SEQ_LEN, body=_encoder,
                                         loop_vars=(tf.constant(0,dtype=tf.int32),h0,h1))

    def _decoder(i, x_t, h_tm1, n_tm1, predictions, samples):
        #stage1
        h_t = enc_rec_unit(pad_voc, h_tm1)
        c_t = enc_map_unit(h_t)
        #stage2
        n_t = dec_rec_unit(x_t, n_tm1, c_t)
        o_t = dec_map_unit(n_t)
        prob = tf.nn.softmax(o_t)
        next_token = tf.cast(tf.reshape(tf.multinomial(tf.log(prob), 1), [BATCH_SIZE]), tf.int32)
        x_tp1 = embed_unit(next_token)
        predictions = predictions.write(i, prob)
        samples = samples.write(i, next_token)
        return i+1, x_tp1, h_t, n_t, predictions, samples
    _, _, _, _, predictions, samples = control_flow_ops.while_loop(cond=lambda i, _1, _2, _3, _4, _5: i < SEQ_LEN,
                                                                   body=_decoder,
                                                                   loop_vars=(tf.constant(0, dtype=tf.int32),
                                                                   start, h_t, n_t, predictions, samples))
    return tf.transpose(samples.stack(), perm=[1,0])


def Generator(inputs_c, inputs_x):
    inputs = tf.transpose(inputs_c, perm=[1,0,2])
    input_token = tensor_array_ops.TensorArray(dtype=tf.float32,size=SEQ_LEN)
    input_token = input_token.unstack(inputs)

    predictions = tensor_array_ops.TensorArray(dtype=tf.float32, size=SEQ_LEN, dynamic_size=False, infer_shape=True)
    samples = tensor_array_ops.TensorArray(dtype=tf.int32, size=SEQ_LEN, dynamic_size=False, infer_shape=True)
    output = tf.transpose(inputs_x, perm=[1,0,2])
    output_token = tensor_array_ops.TensorArray(dtype=tf.float32,size=SEQ_LEN)
    output_token = output_token.unstack(output)

    def _encoder(i, h_tm1, n_tm1):
        #stage1
        x_t = input_token.read(i)
        h_t = enc_rec_unit(x_t, h_tm1)
        c_t = enc_map_unit(h_t)
        #stage2
        n_t = dec_rec_unit(pad_voc, n_tm1, c_t)
        return i+1, h_t, n_t
    _, h_t, n_t= control_flow_ops.while_loop(cond=lambda i, _1, _2: i < SEQ_LEN, body=_encoder,
                                         loop_vars=(tf.constant(0,dtype=tf.int32),h0, h1))

    def _decoder(i, x_t, h_tm1, n_tm1, predictions, samples):
        #stage1
        h_t = enc_rec_unit(pad_voc, h_tm1)
        c_t = enc_map_unit(h_t)
        #stage2
        n_t = dec_rec_unit(x_t, n_tm1, c_t)
        o_t = dec_map_unit(n_t)
        prob = tf.nn.softmax(o_t)
        next_token = tf.cast(tf.reshape(tf.multinomial(tf.log(prob), 1), [BATCH_SIZE]), tf.int32)
        predictions = predictions.write(i, prob)
        samples = samples.write(i, next_token)
        x_tp1 = output_token.read(i)
        return i+1, x_tp1, h_t, n_t, predictions, samples
    _, _, _, _, predictions, samples = control_flow_ops.while_loop(cond=lambda i, _1, _2, _3, _4, _5: i < SEQ_LEN,
                                                         body=_decoder,
                                                         loop_vars=(tf.constant(0, dtype=tf.int32),
                                                         start, h_t, n_t, predictions, samples))
    return tf.transpose(predictions.stack(), perm=[1, 0, 2]), tf.transpose(samples.stack(), perm=[1,0])


def Discriminator(inputs_c, inputs_x):
    inputs = tf.concat([inputs_c, inputs_x], 2)
    inputs = tf.transpose(inputs, perm=[1,0,2])
    input_token = tensor_array_ops.TensorArray(dtype=tf.float32,size=SEQ_LEN)
    input_token = input_token.unstack(inputs)

    def _encoder(i, h_tm1):
        x_t = input_token.read(i)
        h_t = disc_rec_unit(x_t, h_tm1)
        return i+1, h_t
    _, h_t = control_flow_ops.while_loop(cond=lambda i, _1: i < SEQ_LEN, body=_encoder,
                                         loop_vars=(tf.constant(0,dtype=tf.int32),hc))
    return disc_map_unit(h_t)


def inf_train_gen():
    while True:
        np.random.shuffle(lines)
        for i in xrange(0, len(lines)-BATCH_SIZE+1, BATCH_SIZE):
            question = []
            answer = []
            for i in xrange(i,i+BATCH_SIZE,1):
                extra_c, extra_x = zip(*lines[i])
                question.append([wmap[c] for c in extra_c])
                answer.append([wmap[c] for c in extra_x])
            question = np.array(question,dtype='int32')
            answer = np.array(answer,dtype='int32')
            yield list(zip(question, answer))


real_inputs_discrete_c = tf.placeholder(tf.int32, shape=[BATCH_SIZE, SEQ_LEN])
real_inputs_discrete_x = tf.placeholder(tf.int32, shape=[BATCH_SIZE, SEQ_LEN])
poor_inputs_discrete_x = tf.placeholder(tf.int32, shape=[BATCH_SIZE, SEQ_LEN])
start_discrete = tf.to_int32(tf.stack(np.ones([BATCH_SIZE])*wmap['_START_']))

pad_voc  = tf.zeros([BATCH_SIZE, EMB_LEN])
h0 = tf.zeros([BATCH_SIZE, DIM])
h0 = tf.stack([h0, h0])
h1 = tf.zeros([BATCH_SIZE, DIM])
h1 = tf.stack([h1, h1])
hc = tf.zeros([BATCH_SIZE, DIM])
hc = tf.stack([hc, hc])

gen_params = []
embed_unit = Embedding(gen_params)
enc_rec_unit = LSTMEncoder(gen_params)
enc_map_unit = MapEncoder(gen_params)
dec_rec_unit = LSTMDecoder(gen_params)
dec_map_unit = MapDecoder(gen_params)

disc_params = []
disc_rec_unit = LSTMDiscriminator(disc_params)
disc_map_unit = MapDiscriminator(disc_params)

real_embed_c = embed_unit(real_inputs_discrete_c)
real_embed_x = embed_unit(real_inputs_discrete_x)
start = embed_unit(start_discrete)

fake_inputs, fake_int32 = Generator(real_embed_c, real_embed_x)
fake_sample = Sample(real_embed_c, real_embed_x)

real_inputs_c = tf.one_hot(real_inputs_discrete_c, VOC_LEN)
real_inputs_x = tf.one_hot(real_inputs_discrete_x, VOC_LEN)
poor_inputs_x = tf.one_hot(poor_inputs_discrete_x, VOC_LEN)
fake_inputs_x = tf.one_hot(fake_int32, VOC_LEN)

disc_real = Discriminator(real_inputs_c, real_inputs_x)
disc_fake = Discriminator(real_inputs_c, fake_inputs_x)
disc_poor = Discriminator(real_inputs_c, poor_inputs_x) 

disc_cost = -tf.reduce_mean(disc_real) + ( tf.reduce_mean(disc_fake)+tf.reduce_mean(disc_poor) ) / 2

gen_cost_disc = - tf.reduce_mean(disc_fake)
gen_cost_vae  = - tf.reduce_sum( tf.one_hot(tf.reshape(real_inputs_discrete_x, [-1]), VOC_LEN) *
                  tf.log(tf.clip_by_value(tf.reshape(fake_inputs, [-1, VOC_LEN]), 1e-20, 1.0))) / (LAMBDA * SEQ_LEN * BATCH_SIZE)
gen_cost = gen_cost_disc + gen_cost_vae

tf.summary.scalar('disc', disc_cost)
tf.summary.scalar('gen', gen_cost)
tf.summary.scalar('gen-disc', gen_cost_disc)
tf.summary.scalar('gen-vae', gen_cost_vae)

gen_train_op = tf.train.AdamOptimizer(learning_rate=0.0001)
gen_grad, _ = tf.clip_by_global_norm(tf.gradients(gen_cost, gen_params), 5.0)
gen_updates = gen_train_op.apply_gradients(zip(gen_grad, gen_params))

disc_train_op = tf.train.AdamOptimizer(learning_rate=0.0001)
disc_grad, _ = tf.clip_by_global_norm(tf.gradients(disc_cost, disc_params), 5.0)
disc_updates = disc_train_op.apply_gradients(zip(disc_grad, disc_params))
merged_summary_op = tf.summary.merge_all()

start_iter = 0
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter('log/', graph=tf.get_default_graph())

    if os.path.exists("model/checkpoint"):
        ckpt = tf.train.get_checkpoint_state("model")
        start_iter = int(ckpt.model_checkpoint_path.split('chatbot_')[1].split('.model')[0])+1
        tf.train.Saver().restore(session, ckpt.model_checkpoint_path)

    def generate_samples():
        _data_c, _data_x = zip(*gen.next())
        samples = fake_sample.eval(session=session,feed_dict={real_inputs_discrete_c:_data_c,real_inputs_discrete_x:_data_x})
        decoded_samples = []
        for i in xrange(len(samples)):
            decoded_samples.append('Q: '+Parser.decodeData(_data_c[i]))
            decoded_samples.append('A: '+Parser.decodeData(samples[i]))
            decoded_samples.append('\n')
        return decoded_samples

    gen = inf_train_gen()

    for iteration in xrange(ITERS):
        if iteration < start_iter: continue
        start_time = time.time()

        # Train generator
        if iteration > 0:
            _data_c, _data_x = zip(*gen.next())
            _, _poor_x = zip(*gen.next())
            _, summary = session.run([gen_updates, merged_summary_op], feed_dict={real_inputs_discrete_c:_data_c,real_inputs_discrete_x:_data_x,poor_inputs_discrete_x:_poor_x})
            summary_writer.add_summary(summary, iteration)

        # Train critic
        for i in xrange(CRITIC_ITERS):
            _data_c, _data_x = zip(*gen.next())
            _, _poor_x = zip(*gen.next())
            _disc_cost, _, summary = session.run([disc_cost, disc_updates, merged_summary_op], feed_dict={real_inputs_discrete_c:_data_c,real_inputs_discrete_x:_data_x,poor_inputs_discrete_x:_poor_x})
            summary_writer.add_summary(summary, iteration)

        myprint("[Iteration "+ str(iteration) + "] loss=" + str(_disc_cost) + ", time=" + str(time.time()-start_time))

        if iteration % 100 == 99:
            samples = []
            for i in xrange(10):
                samples.extend(generate_samples())

            with open('sample/samples_{}.txt'.format(iteration), 'w') as f:
                for s in samples:
                    s = "".join(s)
                    f.write(s + "\n")

        if iteration % 50000 == 49999:
            tf.train.Saver().save(session, 'model/chatbot_{}.model'.format(iteration), global_step=iteration)
