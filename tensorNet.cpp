#include <algorithm>
//#include "common.h"
#include "tensorNet.h"
#include <sstream>
#include <fstream>

using namespace nvinfer1;


bool TensorNet::LoadNetwork(const char* prototxt_path,
                            const char* model_path,
                            const char* input_blob,
                            const std::vector<std::string>& output_blobs,
                            uint32_t maxBatchSize)
{
    //assert( !prototxt_path || !model_path );

    // attempt to load network from cache before profiling with tensorRT
    std::stringstream gieModelStdStream;
    gieModelStdStream.seekg(0, gieModelStdStream.beg);
    char cache_path[512];
    sprintf(cache_path, "%s.%u.tensorcache", model_path, maxBatchSize);
    printf( "attempting to open cache file %s\n", cache_path);

    std::ifstream cache( cache_path );

    if( !cache )
    {
        printf( "cache file not found, profiling network model\n");

        if( !caffeToTRTModel(prototxt_path, model_path, output_blobs, maxBatchSize, gieModelStdStream) )
        {
            printf("failed to load %s\n", model_path);
            return 0;
        }
        printf( "network profiling complete, writing cache to %s\n", cache_path);
        std::ofstream outFile;
        outFile.open(cache_path);
        outFile << gieModelStdStream.rdbuf();
        outFile.close();
        gieModelStdStream.seekg(0, gieModelStdStream.beg);
        printf( "completed writing cache to %s\n", cache_path);

        infer = createInferRuntime(gLogger);
        /**
         * deserializeCudaEngine can be used to load the serialized CuDA Engine (Plan file).
         * */
        std::cout << "createInference" << std::endl;
        engine = infer->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), &pluginFactory);
        std::cout << "createInference_end" << std::endl;
        printf("Bindings after deserializing:\n");
        for (int bi = 0; bi < engine->getNbBindings(); bi++) {
            if (engine->bindingIsInput(bi) == true) printf("Binding %d (%s): Input.\n",  bi, engine->getBindingName(bi));
            else printf("Binding %d (%s): Output.\n", bi, engine->getBindingName(bi));
        }
    }
    else
    {
        std::cout << "loading network profile from cache..." << std::endl;
        gieModelStdStream << cache.rdbuf();
        cache.close();
        gieModelStdStream.seekg(0, std::ios::end);
        const int modelSize = gieModelStdStream.tellg();
        gieModelStdStream.seekg(0, std::ios::beg);
        void* modelMem = malloc(modelSize);
        gieModelStdStream.read((char*)modelMem, modelSize);

        infer = createInferRuntime(gLogger);
        std::cout << "createInference" << std::endl;
        engine = infer->deserializeCudaEngine(modelMem, modelSize, &pluginFactory);
        //free(modelMem);
        std::cout << "createInference_end" << std::endl;
        printf("Bindings after deserializing:\n");
        for (int bi = 0; bi < engine->getNbBindings(); bi++) {
            if (engine->bindingIsInput(bi) == true) printf("Binding %d (%s): Input.\n",  bi, engine->getBindingName(bi));
            else printf("Binding %d (%s): Output.\n", bi, engine->getBindingName(bi));
        }
    }
}

bool TensorNet::caffeToTRTModel(const char* deployFile,
                                const char* modelFile,
                                const std::vector<std::string>& outputs,
                                unsigned int maxBatchSize,
                                std::ostream& gieModelStdStream)
{
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();
    //    builder->setMinFindIterations(3);	// allow time for TX1 GPU to spin up
    //    builder->setAverageFindIterations(2);
    ICaffeParser* parser = createCaffeParser();
    parser->setPluginFactory(&pluginFactory);

    bool useFp16 = builder->platformHasFastFp16();

    DataType modelDataType = useFp16 ? DataType::kHALF : DataType::kFLOAT;

    std::cout << deployFile <<std::endl;
    std::cout << modelFile <<std::endl;

    const IBlobNameToTensor* blobNameToTensor =	parser->parse(deployFile,
                                                              modelFile,
                                                              *network,
                                                              modelDataType);
    assert(blobNameToTensor != nullptr);
    for (auto& s : outputs) network->markOutput(*blobNameToTensor->find(s.c_str()));

    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(16 << 20);

    if(useFp16)
    {
        builder->setHalf2Mode(true);
    }
    ICudaEngine* engine = builder->buildCudaEngine( *network );
    assert(engine);
    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();
    // serialize the engine, then close everything down
    gieModelStream = engine->serialize();
    if(!gieModelStream)
    {
        std::cout << "failed to serialize CUDA engine" << std::endl;
        return false;
    }
    gieModelStdStream.write((const char*)gieModelStream->data(),gieModelStream->size());
    engine->destroy();
    builder->destroy();
    pluginFactory.destroyPlugin();
    shutdownProtobufLibrary();

    std::cout << "caffeToTRTModel Finished" << std::endl;
    return true;
}

/**
 * This function de-serializes the cuda engine.
 * */
void TensorNet::createInference()
{
    infer = createInferRuntime(gLogger);
    /**
     * deserializeCudaEngine can be used to load the serialized CuDA Engine (Plan file).
     * */
    engine = infer->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), &pluginFactory);

    printf("Bindings after deserializing:\n");
    for (int bi = 0; bi < engine->getNbBindings(); bi++) {
        if (engine->bindingIsInput(bi) == true) printf("Binding %d (%s): Input.\n",  bi, engine->getBindingName(bi));
        else printf("Binding %d (%s): Output.\n", bi, engine->getBindingName(bi));
    }
}

void TensorNet::imageInference(void** buffers, int nbBuffer, int batchSize)
{
    //std::cout << "Came into the image inference method here. "<<std::endl;
    assert( engine->getNbBindings()==nbBuffer);
    IExecutionContext* context = engine->createExecutionContext();
    context->setProfiler(&gProfiler);
    context->execute(batchSize, buffers);
    context->destroy();
}

void TensorNet::timeInference(int iteration, int batchSize)
{
    int inputIdx = 0;
    size_t inputSize = 0;
    void* buffers[engine->getNbBindings()];

    for (int b = 0; b < engine->getNbBindings(); b++)
    {
        DimsCHW dims = static_cast<DimsCHW&&>(engine->getBindingDimensions(b));
        size_t size = batchSize * dims.c() * dims.h() * dims.w() * sizeof(float);
        CHECK(cudaMalloc(&buffers[b], size));

        if(engine->bindingIsInput(b) == true)
        {
            inputIdx = b;
            inputSize = size;
        }
    }

    IExecutionContext* context = engine->createExecutionContext();
    context->setProfiler(&gProfiler);

    CHECK(cudaMemset(buffers[inputIdx], 0, inputSize));

    for (int i = 0; i < iteration;i++) context->execute(batchSize, buffers);

    context->destroy();
    for (int b = 0; b < engine->getNbBindings(); b++) CHECK(cudaFree(buffers[b]));

}

DimsCHW TensorNet::getTensorDims(const char* name)
{
    for (int b = 0; b < engine->getNbBindings(); b++) {
        if( !strcmp( name, engine->getBindingName(b)) )
            return static_cast<DimsCHW&&>(engine->getBindingDimensions(b));
    }
    return DimsCHW{0,0,0};
}

//void TensorNet::getLayerOutput(void** buffers, int nbBuffer, int batchSize)
//{
//    /* *
//     * @TODO: Get the layer with name name in the network
//     * */
//    std::cout << "Came into the image inference method here. "<<std::endl;
//    assert( engine->getNbBindings()==nbBuffer);
//    IExecutionContext* context = engine->createExecutionContext();
//    context->setProfiler(&gProfiler);
//    context->execute( batchSize , buffers);
//
//    context->destroy();
//
//}

void TensorNet::printTimes(int iteration)
{
    gProfiler.printLayerTimes(iteration);
}

void TensorNet::destroy()
{
    pluginFactory.destroyPlugin();
    engine->destroy();
    infer->destroy();
}
