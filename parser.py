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

from os.path import exists
import numpy as np
import collections
import time


global log, wordmap, inv_wordmap
log = open(time.strftime("log/parser_%Y%m%d_%H%M%S"),"w")


def myprint(line):
    print line
    log.write(line+"\n")


def decodeData(code):
    sentence = ''
    for i in range(len(code)):
        if inv_wordmap[code[i]]=='NULL': continue
        if i == 0:
	    sentence = inv_wordmap[code[i]]
        else:
            sentence = sentence + ' ' + inv_wordmap[code[i]]
    return sentence.replace('_undefined','_')


def readCornellPair(dataset, maxline, maxlen, maxvoc):
    global wordmap, inv_wordmap
    fp = open(dataset, 'r')
    spec = ' aaaaapairaaaaa '

    lines = []
    linesZip = []
    for i in range(maxline):
        line = fp.readline()
        if line == '': break
        if line.find(spec)<0 or len(line.split(spec))<2: continue
        lineA = grammarParser(line.split(spec)[0])
        lineB = grammarParser(line.split(spec)[1])
        lineA = lineA + ((" NULL")*(maxlen-len(lineA.split(" "))))
        lineB = lineB + ((" NULL")*(maxlen-len(lineB.split(" "))))
        tokenA = lineA.split(" ")
        tokenB = lineB.split(" ")
        linesZip.append(zip(tokenA[0:maxlen],tokenB[0:maxlen]))
        lines.append(tokenA[0:maxlen])
        lines.append(tokenB[0:maxlen])

    myprint("[ChatBot] read data "+dataset+", maxline="+str(maxline))

    np.random.shuffle(lines)
    counts = collections.Counter(word for line in lines for word in line)
    wordmap = {'_undefined':0,'_START_':1}
    inv_wordmap = ['_undefined','_START_']

    for word,count, in counts.most_common(maxvoc-2):
        if word not in wordmap:
            wordmap[word] = len(inv_wordmap)
            inv_wordmap.append(word)

    filtered_zip = []
    for l in linesZip:
        valid = True
        lineA, lineB = zip(*l)

        filtered_lineA = []
        for word in lineA:
            if word in wordmap:
                filtered_lineA.append(word)
            else:
                filtered_lineA.append('_undefined')

        filtered_lineB = []
        for word in lineB:
            if word in wordmap:
                filtered_lineB.append(word)
            else:
                filtered_lineB.append('_undefined')

        if valid: filtered_zip.append(zip(filtered_lineA,filtered_lineB))
    myprint("[ChatBot] generate word map, maxvoc="+str(maxvoc)+", len="+str(len(filtered_zip)))
    return filtered_zip, wordmap, inv_wordmap


def loadCornellPair(dataset, _wordmap, _inv_wordmap, maxline, maxlen, maxvoc):
    global wordmap, inv_wordmap
    wordmap = _wordmap
    inv_wordmap = _inv_wordmap

    fp = open(dataset, 'r')
    spec = ' aaaaapairaaaaa '

    filtered_zip = []
    for i in range(maxline):
        valid = True
        line = fp.readline()
        if line == '': break
        if line.find(spec)<0 or len(line.split(spec))<2: continue
        lineA = grammarParser(line.split(spec)[0])
        lineB = grammarParser(line.split(spec)[1])
        lineA = lineA + ((" NULL")*(maxlen-len(lineA.split(" "))))
        lineB = lineB + ((" NULL")*(maxlen-len(lineB.split(" "))))
        tokenA = lineA.split(" ")[0:maxlen]
        tokenB = lineB.split(" ")[0:maxlen]

        filtered_lineA = []
        for word in tokenA:
            if word in wordmap:
                filtered_lineA.append(word)
            else:
                filtered_lineA.append('_undefined')

        filtered_lineB = []
        for word in tokenB:
            if word in wordmap:
                filtered_lineB.append(word)
            else:
                filtered_lineB.append('_undefined')

        if valid: filtered_zip.append(zip(filtered_lineA,filtered_lineB))
    myprint("[ChatBot] generate word map, maxvoc="+str(maxvoc)+", len="+str(len(filtered_zip)))
    return filtered_zip


def runtimeLoad(_wordmap, _inv_wordmap):
    global wordmap, inv_wordmap
    wordmap = _wordmap
    inv_wordmap = _inv_wordmap


def runtimeParser(line, maxlen):
    global wordmap, inv_wordmap

    line = grammarParser(line)
    line = line + ((" NULL")*(maxlen-len(line.split(" "))))
    token = line.split(" ")[0:maxlen]

    filtered_line = []
    for word in token:
        if word in wordmap:
            filtered_line.append(word)
        else:
            filtered_line.append('_undefined')
    return filtered_line


def grammarParser(line):
    line = line.replace('<u>','')
    line = line.replace('</u>','')
    line = line.replace('</',' ')
    line = line.replace('<',' ')
    line = line.replace(' \n','')
    line = line.replace('\n','')
    line = line.replace(' \t','')
    line = line.replace('\t','')
    line = line.replace(',',' , ')
    line = line.replace('"',' " ')
    line = line.replace('.',' . ')
    line = line.replace('. . .',' ... ')
    line = line.replace('-',' - ')
    line = line.replace('- -',' -- ')
    line = line.replace('- - -',' --- ')
    line = line.replace('?',' ? ')
    line = line.replace('!',' ! ')
    line = line.replace(';',' ; ')
    line = line.replace('*',' * ')
    line = line.replace(':',' : ')
    line = line.replace('  ',' ')
    line = line.replace('  ',' ')
    line = line.replace('  ',' ')
    line = line.replace('  ',' ')
    line = line.replace('  ',' ')
    line = line.replace('  ',' ')
    if len(line)==0: return ''
    if line[0]==" ":
        output = line[1:len(line)-1]
    else:
        output = line[0:len(line)-1]
    return output.lower()
