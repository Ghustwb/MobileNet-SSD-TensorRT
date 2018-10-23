#include "pluginImplement.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;


/******************************/
// TensorRT utility
/******************************/
class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        if (severity!=Severity::kINFO) std::cout << msg << std::endl;
    }
};

struct Profiler : public IProfiler
{
    typedef std::pair<std::string, float> Record;
    std::vector<Record> mProfile;

    virtual void reportLayerTime(const char* layerName, float ms)
    {
        auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });

        if (record == mProfile.end()) mProfile.push_back(std::make_pair(layerName, ms));
        else record->second += ms;
    }

    void printLayerTimes(const int TIMING_ITERATIONS)
    {
        float totalTime = 0;
        for (size_t i = 0; i < mProfile.size(); i++)
        {
            printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / TIMING_ITERATIONS);
            totalTime += mProfile[i].second;
        }
        printf("Time over all layers: %4.3f\n", totalTime / TIMING_ITERATIONS);
    }
};


/******************************/
// TensorRT Main
/******************************/
class TensorNet
{
public:
    void caffeToTRTModel(const std::string& deployFile,
                         const std::string& modelFile,
                         const std::vector<std::string>& outputs,
                         unsigned int maxBatchSize);
    void createInference();

    void imageInference(void** buffers, int nbBuffer, int batchSize);
    void timeInference(int iteration, int batchSize);

    DimsCHW getTensorDims(const char* name);

//    void getLayerOutput(const char* name);

    void printTimes(int iteration);
    void destroy();

private:

    PluginFactory pluginFactory;
    IHostMemory *gieModelStream{nullptr};

    IRuntime* infer;
    ICudaEngine* engine;

    Logger gLogger;
    Profiler gProfiler;

};


//#endif

