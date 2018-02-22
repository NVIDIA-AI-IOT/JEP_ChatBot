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

import parser as Parser
import numpy as np
import pickle
import sys

sys.path.insert(0, 'src/')
import tensorNet

SEQ_LEN = 32

if len(sys.argv) < 3:
    print 'Usage: python chatbot.py [Text pickle] [TRT Model]'
    sys.exit(0)


with open(sys.argv[1]) as f:
    wmap, iwmap = pickle.load(f)
Parser.runtimeLoad(wmap, iwmap)
print '[ChatBot] load word map from '+sys.argv[1]


engine = tensorNet.createTrtFromUFF(sys.argv[2])
tensorNet.prepareBuffer(engine)
print '[ChatBot] create tensorrt engine from '+sys.argv[2]


while True:
    b = raw_input('\n\n\x1b[1;105;97m'+'Please write your question(q for quite):'+'\x1b[0m')
    if b=='q':
        print 'Bye Bye!!'
        break
    elif len(b)>0:
        raw = Parser.runtimeParser(b+' ', SEQ_LEN)
        question = []
        question.append([wmap[c] for c in raw])

        _input  = np.array(question[0], np.int32)
        _output = np.zeros([SEQ_LEN], np.int32)
        tensorNet.inference(engine, _input, _output)
        print 'Q: '+'\x1b[1;39;94m'+Parser.decodeData(question[0])+'\x1b[0m'
        print 'A: '+'\x1b[1;39;92m'+Parser.decodeData(_output.tolist())+'\x1b[0m'
