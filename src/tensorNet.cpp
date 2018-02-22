/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "tensorNet.h"
#include "NvUffParser.h"
#include <cuda_runtime.h>

using namespace nvuffparser;
using namespace nvinfer1;

static Logger gLogger;
static std::vector<void*> buffers;

#define MAX_WORKSPACE (1 << 30)


size_t getBufferSize(Dims d, DataType t)
{
    size_t size = 1;
    for(size_t i=0; i<d.nbDims; i++) size*= d.d[i];

    switch (t) {
        case DataType::kFLOAT: return size*4;
        case DataType::kHALF: return size*2;
        case DataType::kINT8: return size*1;
    }
    assert(0);
    return 0;
}

ICudaEngine* createTrtFromUFF(char* modelpath)
{
    auto parser = createUffParser();

    parser->registerInput("enc_text", DimsCHW(1, VOC_LEN, 1));
    parser->registerInput("dec_text", DimsCHW(1, VOC_LEN, 1));
    parser->registerInput("h0_in", DimsCHW(1, DIM, 1));
    parser->registerInput("c0_in", DimsCHW(1, DIM, 1));
    parser->registerInput("h1_in", DimsCHW(1, DIM, 1));
    parser->registerInput("c1_in", DimsCHW(1, DIM, 1));

    parser->registerOutput("h0_out");
    parser->registerOutput("c0_out");
    parser->registerOutput("h1_out");
    parser->registerOutput("c1_out");
    parser->registerOutput("final_output");

    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();

    if (!parser->parse(modelpath, *network, nvinfer1::DataType::kFLOAT)) {
        std::cout << "[ChatBot] Fail to parse UFF model " << modelpath << std::endl;
        exit(0);
    }

    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(MAX_WORKSPACE);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (!engine) {
        std::cout << "[ChatBot] Unable to create engine" << std::endl;
        exit(0);
    }

    network->destroy();
    builder->destroy();
    parser->destroy();

    std::cout << "[ChatBot] Successfully create TensorRT engine from file " << modelpath << std::endl;

    return engine;
}

void prepareBuffer(ICudaEngine* engine)
{
    if(!engine) {
        std::cout << "[ChatBot] Invaild engine. Please remember to create engine first." << std::endl;
        exit(0);
    }
    IExecutionContext* context = engine->createExecutionContext();

    int nbBindings = engine->getNbBindings();
    assert(nbBindings == 11);

    buffers.clear();
    buffers.reserve(nbBindings);

    for( int i=0; i<nbBindings; i++ ) {
        cudaMallocManaged(&buffers[i],  getBufferSize(engine->getBindingDimensions(i),  engine->getBindingDataType(i)));
    }

    buffers[6] = buffers[2];   //h0_out=h0_in
    buffers[7] = buffers[3];   //c0_out=c0_in
    buffers[8] = buffers[4];   //h1_out=h1_in
    buffers[9] = buffers[5];   //c1_out=c1_in

    std::cout << "[ChatBot] Successfully create binding buffer" << std::endl;
}

void inference(ICudaEngine* engine,
               int dim_in,  int* data_in,
               int dim_out, int* data_out)
{
    if(!engine) {
        std::cout << "[ChatBot] Invaild engine. Please remember to create engine first." << std::endl;
        exit(0);
    }
    IExecutionContext* context = engine->createExecutionContext();

    for (int i=0; i<6; i++) {
        size_t size = getBufferSize(engine->getBindingDimensions(i), engine->getBindingDataType(i));

        float* buffer = (float*)buffers[i];
        for(int j=0; j<size; j++) buffer[j] = 0;
    }

    float* _enc_text = (float*)buffers[0];
    float* _dec_text = (float*)buffers[1];
    float* _final  = (float*)buffers[10];

    for( int i=0; i<SEQ_LEN; i++) {
        _dec_text[data_in[i]] = 1;

        cudaThreadSynchronize();
        context->execute(1, &buffers[0]);
        cudaThreadSynchronize();

        _dec_text[data_in[i]] = 0;
    }

    int pre_token = 1;
    for(int i=0; i<SEQ_LEN; i++) {
        _enc_text[pre_token] = 1;

        cudaThreadSynchronize();
        context->execute(1, &buffers[0]);
        cudaThreadSynchronize();

        int max_index = -1;
        float max_value = -100;
        for(int j=0; j<VOC_LEN; j++) {
            if( _final[j] > max_value ) {
                max_value = _final[j];
                max_index = j;
            }
        }
        _enc_text[pre_token] = 0;
        pre_token = max_index;
        data_out[i] = pre_token;
    }
}
