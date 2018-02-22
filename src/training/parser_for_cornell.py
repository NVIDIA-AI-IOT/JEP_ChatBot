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

import numpy as np

input_conv = 'movie_conversations.txt'
input_line = 'movie_lines.txt'
output = 'cornell_pair'

input_conv_fp = open(input_conv,'r')
input_line_fp = open(input_line,'r')
fp = open(output,'w')

conversation = input_conv_fp.readlines()
lines = input_line_fp.readlines()


for pairs in conversation:
    l = pairs.split('[')[1].split(']')[0]
    print l
    p = []
    for t in l.split('\''):
        if t.find('L') < 0: continue
        p.append(t)

    if len(p) < 2: continue
    for i in range(len(p)-1):
        for line in lines:
            if p[i] in line:
                a = line.split(' +++$+++ ')[4].replace('\n','')
                break
        for line in lines:
            if p[i+1] in line:
                b = line.split(' +++$+++ ')[4].replace('\n','')
	        break

        if len(a) < 4: continue
        if len(b) < 4: continue
        fp.write(a+' aaaaapairaaaaa '+b+'\n')


fp.close()
input_line_fp.close()
input_conv_fp.close()
