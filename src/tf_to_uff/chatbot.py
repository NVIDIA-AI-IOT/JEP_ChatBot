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

from tensorrt.parsers import uffparser
import pycuda.driver as cuda
import tensorrt as trt
import parser as Parser
import numpy as np
import uff
import time
import model
import pickle
import sys

MAX_BATCHSIZE = 1
SEQ_LEN = 32
DIM = 128
VOC_LEN = 4096


def createTrtFromUFF(modelpath):
    MAX_WORKSPACE = 1 << 30
    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)

    parser = uffparser.create_uff_parser()
    parser.register_input("enc_text", (1, VOC_LEN, 1), 0)
    parser.register_input("dec_text", (1, VOC_LEN, 1), 1)
    parser.register_input("h0_in", (1, DIM, 1), 2)
    parser.register_input("c0_in", (1, DIM, 1), 3)
    parser.register_input("h1_in", (1, DIM, 1), 4)
    parser.register_input("c1_in", (1, DIM, 1), 5)

    parser.register_output("h0_out")
    parser.register_output("c0_out")
    parser.register_output("h1_out")
    parser.register_output("c1_out")
    parser.register_output("final_output")

    engine = trt.utils.uff_file_to_trt_engine(G_LOGGER, modelpath, parser, MAX_BATCHSIZE, MAX_WORKSPACE, trt.infer.DataType.FLOAT)
    print '[ChatBot] Successfully create TensorRT engine from file '+modelpath
    return engine


def inference(question):
    # preparison
    _enc_text = np.zeros([MAX_BATCHSIZE, VOC_LEN], np.float32)
    _dec_text = np.zeros([MAX_BATCHSIZE, VOC_LEN], np.float32)
    cuda.memcpy_htod_async(d_in_h0, _h0, stream)
    cuda.memcpy_htod_async(d_in_c0, _c0, stream)
    cuda.memcpy_htod_async(d_in_h1, _h1, stream)
    cuda.memcpy_htod_async(d_in_c1, _c1, stream)

    # encoder
    for i in range(SEQ_LEN):
        _enc_text[0][question[i]] = 1
        cuda.memcpy_htod_async(d_in_dec, _enc_text, stream)
        cuda.memcpy_htod_async(d_in_enc, _dec_text, stream)
        time.sleep(0.001)
        context.enqueue(1, bindings, stream.handle, None)

        _enc_text[0][question[i]] = 0

    # decoder
    _enc_text = np.zeros([MAX_BATCHSIZE, VOC_LEN], np.float32)
    answer = np.zeros([SEQ_LEN], np.int32)
    pre_token = wmap['_START_']

    for i in range(SEQ_LEN):
        _dec_text[0][pre_token] = 1
        cuda.memcpy_htod_async(d_in_dec, _enc_text, stream)
        cuda.memcpy_htod_async(d_in_enc, _dec_text, stream)
        context.enqueue(1, bindings, stream.handle, None)

        cuda.memcpy_dtoh_async(output_ot, d_ot_ot, stream)
        _dec_text[0][pre_token] = 0
        pre_token = np.argmax(output_ot)
        answer[i] = pre_token

    return answer


if len(sys.argv) < 3:
    print 'Usage: python chatbot.py [Text pickle] [TRT Model]'
    sys.exit(0)

with open(sys.argv[1]) as f:
    wmap, iwmap = pickle.load(f)
Parser.runtimeLoad(wmap, iwmap)
print '[ChatBot] load word map from '+sys.argv[1]

engine = createTrtFromUFF(sys.argv[2])
context = engine.create_execution_context()
print '[ChatBot] create tensorrt engine from '+sys.argv[2]

_enc_text = np.zeros([MAX_BATCHSIZE, VOC_LEN], np.float32)
_dec_text = np.zeros([MAX_BATCHSIZE, VOC_LEN], np.float32)
_h0 = np.zeros([MAX_BATCHSIZE, DIM], np.float32)
_c0 = np.zeros([MAX_BATCHSIZE, DIM], np.float32)
_h1 = np.zeros([MAX_BATCHSIZE, DIM], np.float32)
_c1 = np.zeros([MAX_BATCHSIZE, DIM], np.float32)

dims_enc = engine.get_binding_dimensions(0).to_DimsCHW()
dims_dec = engine.get_binding_dimensions(1).to_DimsCHW()
dims_h0 = engine.get_binding_dimensions(2).to_DimsCHW()
dims_c0 = engine.get_binding_dimensions(3).to_DimsCHW()
dims_h1 = engine.get_binding_dimensions(4).to_DimsCHW()
dims_c1 = engine.get_binding_dimensions(5).to_DimsCHW()
dims_ot = engine.get_binding_dimensions(10).to_DimsCHW()

output_ot = cuda.pagelocked_empty(dims_ot.C() * dims_ot.H() * dims_ot.W() * MAX_BATCHSIZE, dtype=np.float32)

d_in_enc = cuda.mem_alloc(MAX_BATCHSIZE * dims_enc.C() * dims_enc.H() * dims_enc.W() * _enc_text.dtype.itemsize)
d_in_dec = cuda.mem_alloc(MAX_BATCHSIZE * dims_dec.C() * dims_dec.H() * dims_dec.W() * _dec_text.dtype.itemsize)
d_in_h0  = cuda.mem_alloc(MAX_BATCHSIZE * dims_h0.C()  * dims_h0.H()  * dims_h0.W()  * _h0.dtype.itemsize)
d_in_c0  = cuda.mem_alloc(MAX_BATCHSIZE * dims_c0.C()  * dims_c0.H()  * dims_c0.W()  * _c0.dtype.itemsize)
d_in_h1  = cuda.mem_alloc(MAX_BATCHSIZE * dims_h1.C()  * dims_h1.H()  * dims_h1.W()  * _h1.dtype.itemsize)
d_in_c1  = cuda.mem_alloc(MAX_BATCHSIZE * dims_c1.C()  * dims_c1.H()  * dims_c1.W()  * _c1.dtype.itemsize)
d_ot_ot  = cuda.mem_alloc(MAX_BATCHSIZE * dims_ot.C()  * dims_ot.H()  * dims_ot.W()  * output_ot.dtype.itemsize)

bindings = [int(d_in_enc), int(d_in_dec), int(d_in_h0), int(d_in_c0), int(d_in_h1), int(d_in_c1),
            int(d_in_h0),  int(d_in_c0),  int(d_in_h1), int(d_in_c1),  int(d_ot_ot)]
stream = cuda.Stream()

while True:
    b = raw_input('\n\n\x1b[1;105;97m'+'Please write your question(q for quite):'+'\x1b[0m')
    if b=='q':
        print 'Bye Bye!!'
        break
    elif len(b)>0:
        raw = Parser.runtimeParser(b+' ', SEQ_LEN)
        question = []
        question.append([wmap[c] for c in raw])
        question = np.array(question,dtype='int32')
        print 'Q: '+'\x1b[1;39;94m'+Parser.decodeData(question[0])+'\x1b[0m'
        print 'A: '+'\x1b[1;39;92m'+Parser.decodeData(inference(question[0]))+'\x1b[0m'
