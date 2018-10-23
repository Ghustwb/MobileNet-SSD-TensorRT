#include "pluginImplement.h"
#include "mathFunctions.h"
#include <vector>
#include <algorithm>

/******************************/
// Softmax Plugin Layer
/******************************/
//The code is not publicly available,, and you need to implement it yourself.

/******************************/
// Concat Plugin Layer
/******************************/
/*
ConcatPlugin::ConcatPlugin(int axis, const void* buffer, size_t size)
{
    assert(size == (18*sizeof(int)));
    const int* d = reinterpret_cast<const int*>(buffer);

    dimsConv4_3 = DimsCHW{d[0], d[1], d[2]};
    dimsFc7 = DimsCHW{d[3], d[4], d[5]};
    dimsConv6 = DimsCHW{d[6], d[7], d[8]};
    dimsConv7 = DimsCHW{d[9], d[10], d[11]};
    dimsConv8 = DimsCHW{d[12], d[13], d[14]};
    dimsConv9 = DimsCHW{d[15], d[16], d[17]};
    _axis = axis;
}

Dims ConcatPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(nbInputDims == 6);
    if(_axis == 1)
    {
        top_concat_axis = inputs[0].d[0] + inputs[1].d[0] + inputs[2].d[0] + inputs[3].d[0] + inputs[4].d[0] + inputs[5].d[0];
        return DimsCHW(top_concat_axis, 1, 1);
    }
    else if(_axis == 2)
    {
        top_concat_axis = inputs[0].d[1] + inputs[1].d[1] + inputs[2].d[1] + inputs[3].d[1] + inputs[4].d[1] + inputs[5].d[1];
        return DimsCHW(2, top_concat_axis, 1);
    }
    else
    {//_param.concat_axis == 3
        return DimsCHW(0, 0, 0);
    }
}

int ConcatPlugin::initialize()
{
    inputs_size = 6;//6个bottom层
    if(_axis == 1)//c
    {
        top_concat_axis = dimsConv4_3.c() + dimsFc7.c() + dimsConv6.c() + dimsConv7.c() + dimsConv8.c() + dimsConv9.c();
        bottom_concat_axis[0] = dimsConv4_3.c(); bottom_concat_axis[1] = dimsFc7.c(); bottom_concat_axis[2] = dimsConv6.c();
        bottom_concat_axis[3] = dimsConv7.c(); bottom_concat_axis[4] = dimsConv8.c(); bottom_concat_axis[5] = dimsConv9.c();

        concat_input_size_[0] = dimsConv4_3.h() * dimsConv4_3.w();  concat_input_size_[1] = dimsFc7.h() * dimsFc7.w();
        concat_input_size_[2] = dimsConv6.h() * dimsConv6.w();  concat_input_size_[3] = dimsConv7.h() * dimsConv7.w();
        concat_input_size_[4] = dimsConv8.h() * dimsConv8.w();  concat_input_size_[5] = dimsConv9.h() * dimsConv9.w();

        num_concats_[0] = dimsConv4_3.c(); num_concats_[1] = dimsFc7.c(); num_concats_[2] = dimsConv6.c();
        num_concats_[3] = dimsConv7.c(); num_concats_[4] = dimsConv8.c(); num_concats_[5] = dimsConv9.c();
    }
    else if(_axis == 2)
    {//h
        top_concat_axis = dimsConv4_3.h() + dimsFc7.h() + dimsConv6.h() + dimsConv7.h() + dimsConv8.h() + dimsConv9.h();
        bottom_concat_axis[0] = dimsConv4_3.h(); bottom_concat_axis[1] = dimsFc7.h(); bottom_concat_axis[2] = dimsConv6.h();
        bottom_concat_axis[3] = dimsConv7.h(); bottom_concat_axis[4] = dimsConv8.h(); bottom_concat_axis[5] = dimsConv9.h();

        concat_input_size_[0] = dimsConv4_3.w(); concat_input_size_[1] = dimsFc7.w(); concat_input_size_[2] = dimsConv6.w();
        concat_input_size_[3] = dimsConv7.w(); concat_input_size_[4] = dimsConv8.w(); concat_input_size_[5] = dimsConv9.w();

        num_concats_[0] = dimsConv4_3.c() * dimsConv4_3.h();  num_concats_[1] = dimsFc7.c() * dimsFc7.h();
        num_concats_[2] = dimsConv6.c() * dimsConv6.h();  num_concats_[3] = dimsConv7.c() * dimsConv7.h();
        num_concats_[4] = dimsConv8.c() * dimsConv8.h();  num_concats_[5] = dimsConv9.c() * dimsConv9.h();

    }
    else
    {//_param.concat_axis == 3 , w
        top_concat_axis = dimsConv4_3.w() + dimsFc7.w() + dimsConv6.w() + dimsConv7.w() + dimsConv8.w() + dimsConv9.w();
        bottom_concat_axis[0] = dimsConv4_3.w(); bottom_concat_axis[1] = dimsFc7.w(); bottom_concat_axis[2] = dimsConv6.w();
        bottom_concat_axis[3] = dimsConv7.w(); bottom_concat_axis[4] = dimsConv8.w(); bottom_concat_axis[5] = dimsConv9.w();

        concat_input_size_[0] = 1; concat_input_size_[1] = 1; concat_input_size_[2] = 1;
        concat_input_size_[3] = 1; concat_input_size_[4] = 1; concat_input_size_[5] = 1;
        return 0;
    }
    return 0;
}

void ConcatPlugin::terminate()
{
    //CUDA_CHECK(cudaFree(scale_data));
    delete[] bottom_concat_axis;
    delete[] concat_input_size_;
    delete[] num_concats_;
}


int ConcatPlugin::enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream)
{
    float *top_data = reinterpret_cast<float*>(outputs[0]);
    int offset_concat_axis = 0;
    const bool kForward = true;
    for (int i = 0; i < inputs_size; ++i) {
        const float *bottom_data = reinterpret_cast<const float*>(inputs[i]);

        const int nthreads = num_concats_[i] * concat_input_size_[i];
        //const int nthreads = bottom_concat_size * num_concats_[i];
        ConcatLayer(nthreads, bottom_data, kForward, num_concats_[i], concat_input_size_[i], top_concat_axis, bottom_concat_axis[i], offset_concat_axis, top_data, stream);

        offset_concat_axis += bottom_concat_axis[i];
    }

    return 0;
}

size_t ConcatPlugin::getSerializationSize()
{
    return 18*sizeof(int);
}

void ConcatPlugin::serialize(void* buffer)
{
    int* d = reinterpret_cast<int*>(buffer);
    d[0] = dimsConv4_3.c(); d[1] = dimsConv4_3.h(); d[2] = dimsConv4_3.w();
    d[3] = dimsFc7.c(); d[4] = dimsFc7.h(); d[5] = dimsFc7.w();
    d[6] = dimsConv6.c(); d[7] = dimsConv6.h(); d[8] = dimsConv6.w();
    d[9] = dimsConv7.c(); d[10] = dimsConv7.h(); d[11] = dimsConv7.w();
    d[12] = dimsConv8.c(); d[13] = dimsConv8.h(); d[14] = dimsConv8.w();
    d[15] = dimsConv9.c(); d[16] = dimsConv9.h(); d[17] = dimsConv9.w();
}

void ConcatPlugin::configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)
{
    dimsConv4_3 = DimsCHW{inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]};
    dimsFc7 = DimsCHW{inputs[1].d[0], inputs[1].d[1], inputs[1].d[2]};
    dimsConv6 = DimsCHW{inputs[2].d[0], inputs[2].d[1], inputs[2].d[2]};
    dimsConv7 = DimsCHW{inputs[3].d[0], inputs[3].d[1], inputs[3].d[2]};
    dimsConv8 = DimsCHW{inputs[4].d[0], inputs[4].d[1], inputs[4].d[2]};
    dimsConv9 = DimsCHW{inputs[5].d[0], inputs[5].d[1], inputs[5].d[2]};
}
*/

/******************************/
// PluginFactory
/******************************/
nvinfer1::IPlugin* PluginFactory::createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights)
{
    assert(isPlugin(layerName));
    if (!strcmp(layerName, "conv11_mbox_loc_perm"))
    {
        assert(mConv11_mbox_loc_perm_layer.get() == nullptr);
        mConv11_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mConv11_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv11_mbox_conf_perm"))
    {
        assert(mConv11_mbox_conf_perm_layer.get() == nullptr);
        mConv11_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mConv11_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv13_mbox_loc_perm"))
    {
        assert(mConv13_mbox_loc_perm_layer.get() == nullptr);
        mConv13_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mConv13_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv13_mbox_conf_perm"))
    {
        assert(mConv13_mbox_conf_perm_layer.get() == nullptr);
        mConv13_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mConv13_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv14_2_mbox_loc_perm"))
    {
        assert(mConv14_2_mbox_loc_perm_layer.get() == nullptr);
        mConv14_2_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mConv14_2_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv14_2_mbox_conf_perm"))
    {
        assert(mConv14_2_mbox_conf_perm_layer.get() == nullptr);
        mConv14_2_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mConv14_2_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv15_2_mbox_loc_perm"))
    {
        assert(mConv15_2_mbox_loc_perm_layer.get() == nullptr);
        mConv15_2_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mConv15_2_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv15_2_mbox_conf_perm"))
    {
        assert(mConv15_2_mbox_conf_perm_layer.get() == nullptr);
        mConv15_2_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mConv15_2_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv16_2_mbox_loc_perm"))
    {
        assert(mConv16_2_mbox_loc_perm_layer.get() == nullptr);
        mConv16_2_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mConv16_2_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv16_2_mbox_conf_perm"))
    {
        assert(mConv16_2_mbox_conf_perm_layer.get() == nullptr);
        mConv16_2_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mConv16_2_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv17_2_mbox_conf_perm"))
    {
        assert(mConv17_2_mbox_conf_perm_layer.get() == nullptr);
        mConv17_2_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mConv17_2_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv17_2_mbox_loc_perm"))
    {
        assert(mConv17_2_mbox_loc_perm_layer.get() == nullptr);
        mConv17_2_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin({{0, 2, 3, 1}}), nvPluginDeleter);
        return mConv17_2_mbox_loc_perm_layer.get();
    }

    else if (!strcmp(layerName, "conv11_mbox_priorbox"))
    {
        assert(mConv11_mbox_priorbox_layer.get() == nullptr);
        //参数按照原来的prototxt中的prior_box_param设置
        PriorBoxParameters params;
        float minsize[1] = {60},aspect_ratio[2] = {1.0,2.0};
        params.minSize = minsize;
        params.aspectRatios = aspect_ratio;
        params.numMinSize = 1;
        params.numAspectRatios = 2;
        params.maxSize = nullptr;
        params.numMaxSize = 0;
        params.flip = true;
        params.clip = false;
        params.variance[0] = 0.1;
        params.variance[1] = 0.1;
        params.variance[2] = 0.2;
        params.variance[3] = 0.2;
        params.imgH = 0;
        params.imgW = 0;
        params.stepH = 0;
        params.stepW = 0;
        params.offset = 0.5;
        mConv11_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(params), nvPluginDeleter);
        return mConv11_mbox_priorbox_layer.get();
    }
    else if (!strcmp(layerName, "conv13_mbox_priorbox"))
    {
        assert(mConv13_mbox_priorbox_layer.get() == nullptr);
        PriorBoxParameters params;
        float minsize[1] = {105}, maxsize[1] = {150}, aspect_ratio[3] = {1.0,2.0,3.0};
        params.minSize = minsize;
        params.numMinSize = 1;
        params.maxSize = maxsize;
        params.numMaxSize = 1;
        params.aspectRatios = aspect_ratio;
        params.numAspectRatios = 3;

        params.flip = true;
        params.clip = false;
        params.variance[0] = 0.1;
        params.variance[1] = 0.1;
        params.variance[2] = 0.2;
        params.variance[3] = 0.2;
        params.imgH = 0;
        params.imgW = 0;
        params.stepH = 0;
        params.stepW = 0;
        params.offset = 0.5;
        mConv13_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(params), nvPluginDeleter);
        return mConv13_mbox_priorbox_layer.get();
    }
    else if (!strcmp(layerName, "conv14_2_mbox_priorbox"))
    {
        assert(mConv14_2_mbox_priorbox_layer.get() == nullptr);
        PriorBoxParameters params;
        float minsize[1] = {150}, maxsize[1] = {195}, aspect_ratio[3] = {1.0,2.0, 3.0};
        params.minSize = minsize;
        params.numMinSize = 1;
        params.maxSize = maxsize;
        params.numMaxSize = 1;
        params.aspectRatios = aspect_ratio;
        params.numAspectRatios = 3;

        params.flip = true;
        params.clip = false;
        params.variance[0] = 0.1;
        params.variance[1] = 0.1;
        params.variance[2] = 0.2;
        params.variance[3] = 0.2;
        params.imgH = 0;
        params.imgW = 0;
        params.stepH = 0;
        params.stepW = 0;
        params.offset = 0.5;
        mConv14_2_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(params), nvPluginDeleter);
        return mConv14_2_mbox_priorbox_layer.get();
    }
    else if (!strcmp(layerName, "conv15_2_mbox_priorbox"))
    {
        assert(mConv15_2_mbox_priorbox_layer.get() == nullptr);
        PriorBoxParameters params;
        float minsize[1] = {195}, maxsize[3] = {240}, aspect_ratio[3] = {1.0,2.0, 3.0};
        params.minSize = minsize;
        params.numMinSize = 1;
        params.maxSize = maxsize;
        params.numMaxSize = 1;
        params.aspectRatios = aspect_ratio;
        params.numAspectRatios = 3;

        params.flip = true;
        params.clip = false;
        params.variance[0] = 0.1;
        params.variance[1] = 0.1;
        params.variance[2] = 0.2;
        params.variance[3] = 0.2;
        params.imgH = 0;
        params.imgW = 0;
        params.stepH = 0;
        params.stepW = 0;
        params.offset = 0.5;
        mConv15_2_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(params), nvPluginDeleter);
        return mConv15_2_mbox_priorbox_layer.get();
    }
    else if (!strcmp(layerName, "conv16_2_mbox_priorbox"))
    {
        assert(mConv16_2_mbox_priorbox_layer.get() == nullptr);
        PriorBoxParameters params;
        float minsize[1] = {240}, maxsize[1] = {285}, aspect_ratio[3] = {1.0,2.0, 3.0};
        params.minSize = minsize;
        params.numMinSize = 1;
        params.maxSize = maxsize;
        params.numMaxSize = 1;
        params.aspectRatios = aspect_ratio;
        params.numAspectRatios = 3;

        params.flip = true;
        params.clip = false;
        params.variance[0] = 0.1;
        params.variance[1] = 0.1;
        params.variance[2] = 0.2;
        params.variance[3] = 0.2;
        params.imgH = 0;
        params.imgW = 0;
        params.stepH = 0;
        params.stepW = 0;
        params.offset = 0.5;
        mConv16_2_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(params), nvPluginDeleter);
        return mConv16_2_mbox_priorbox_layer.get();
    }
    else if (!strcmp(layerName, "conv17_2_mbox_priorbox"))
    {
        assert(mConv17_2_mbox_priorbox_layer.get() == nullptr);
        PriorBoxParameters params;
        float minsize[1] = {285}, maxsize[1] = {300}, aspect_ratio[3] = {1.0,2.0, 3.0};
        params.minSize = minsize;
        params.numMinSize = 1;
        params.maxSize = maxsize;
        params.numMaxSize = 1;
        params.aspectRatios = aspect_ratio;
        params.numAspectRatios = 3;

        params.flip = true;
        params.clip = false;
        params.variance[0] = 0.1;
        params.variance[1] = 0.1;
        params.variance[2] = 0.2;
        params.variance[3] = 0.2;
        params.imgH = 0;
        params.imgW = 0;
        params.stepH = 0;
        params.stepW = 0;
        params.offset = 0.5;
        mConv17_2_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(params), nvPluginDeleter);
        return mConv17_2_mbox_priorbox_layer.get();
    }

    else if (!strcmp(layerName, "mbox_loc"))
    {
        assert(mBox_loc_layer.get() == nullptr);
        mBox_loc_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        //mBox_loc_layer = std::unique_ptr<ConcatPlugin>(new ConcatPlugin(1));
        return mBox_loc_layer.get();
    }
    else if (!strcmp(layerName, "mbox_conf"))
    {
        assert(mBox_conf_layer.get() == nullptr);
        mBox_conf_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(1, true), nvPluginDeleter);
        //mBox_conf_layer = std::unique_ptr<ConcatPlugin>(new ConcatPlugin(1));
        return mBox_conf_layer.get();
    }
    else if (!strcmp(layerName, "mbox_priorbox"))
    {
        assert(mBox_priorbox_layer.get() == nullptr);
        mBox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(2, true), nvPluginDeleter);
        //mBox_priorbox_layer = std::unique_ptr<ConcatPlugin>(new ConcatPlugin(2));
        return mBox_priorbox_layer.get();
    }
    //flatten
    else if (!strcmp(layerName, "conv11_mbox_conf_flat"))
    {
        assert(mConv11_mbox_conf_flat_layer.get() == nullptr);
        mConv11_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mConv11_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv13_mbox_conf_flat"))
    {
        assert(mConv13_mbox_conf_flat_layer.get() == nullptr);
        mConv13_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mConv13_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv14_2_mbox_conf_flat"))
    {
        assert(mConv14_2_mbox_conf_flat_layer.get() == nullptr);
        mConv14_2_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mConv14_2_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv15_2_mbox_conf_flat"))
    {
        assert(mConv15_2_mbox_conf_flat_layer.get() == nullptr);
        mConv15_2_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mConv15_2_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv16_2_mbox_conf_flat"))
    {
        assert(mConv16_2_mbox_conf_flat_layer.get() == nullptr);
        mConv16_2_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mConv16_2_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv17_2_mbox_conf_flat"))
    {
        assert(mConv17_2_mbox_conf_flat_layer.get() == nullptr);
        mConv17_2_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mConv17_2_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv11_mbox_loc_flat"))
    {
        assert(mConv11_mbox_loc_flat_layer.get() == nullptr);
        mConv11_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mConv11_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv13_mbox_loc_flat"))
    {
        assert(mConv13_mbox_loc_flat_layer.get() == nullptr);
        mConv13_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mConv13_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv14_2_mbox_loc_flat"))
    {
        assert(mConv14_2_mbox_loc_flat_layer.get() == nullptr);
        mConv14_2_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mConv14_2_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv15_2_mbox_loc_flat"))
    {
        assert(mConv15_2_mbox_loc_flat_layer.get() == nullptr);
        mConv15_2_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mConv15_2_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv16_2_mbox_loc_flat"))
    {
        assert(mConv16_2_mbox_loc_flat_layer.get() == nullptr);
        mConv16_2_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mConv16_2_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv17_2_mbox_loc_flat"))
    {
        assert(mConv17_2_mbox_loc_flat_layer.get() == nullptr);
        mConv17_2_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mConv17_2_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "mbox_conf_flatten"))
    {
        assert(mMbox_conf_flat_layer.get() == nullptr);
        mMbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer());
        return mMbox_conf_flat_layer.get();
    }

    //reshape
    else if (!strcmp(layerName, "mbox_conf_reshape"))
    {
        assert(mMbox_conf_reshape.get() == nullptr);
        assert(nbWeights == 0 && weights == nullptr);
        mMbox_conf_reshape = std::unique_ptr<Reshape<5>>(new Reshape<5>());
        return mMbox_conf_reshape.get();
    }
    //softmax layer
    else if (!strcmp(layerName, "mbox_conf_softmax"))
    {
        assert(mPluginSoftmax == nullptr);
        assert(nbWeights == 0 && weights == nullptr);
        mPluginSoftmax = std::unique_ptr<SoftmaxPlugin>(new SoftmaxPlugin());
        return mPluginSoftmax.get();
    }
    else if (!strcmp(layerName, "detection_out"))
    {
        assert(mDetection_out.get() == nullptr);
        DetectionOutputParameters params;

        params.backgroundLabelId = 0;
        //params.codeType = CodeTypeSSD::CENTER_SIZE;
        params.codeType = CodeType_t::CENTER_SIZE;
        params.keepTopK = 100;
        params.shareLocation = true;
        params.varianceEncodedInTarget = false;
        params.topK = 100;
        params.nmsThreshold = 0.45;
        params.numClasses = 5;
//        params.inputOrder[0] = 0;
//        params.inputOrder[1] = 1;
//        params.inputOrder[2] = 2;
        params.confidenceThreshold = 0.4;

        mDetection_out = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDDetectionOutputPlugin(params), nvPluginDeleter);

        return mDetection_out.get();
    }

    else
    {
        std::cout << "not found  " << layerName << std::endl;
        assert(0);
        return nullptr;
    }
}

IPlugin* PluginFactory::createPlugin(const char* layerName, const void* serialData, size_t serialLength)
{
    assert(isPlugin(layerName));
    if (!strcmp(layerName, "conv11_mbox_loc_perm"))
    {
        assert(mConv11_mbox_loc_perm_layer.get() == nullptr);
        mConv11_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return mConv11_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv11_mbox_conf_perm"))
    {
        assert(mConv11_mbox_conf_perm_layer.get() == nullptr);
        mConv11_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return mConv11_mbox_conf_perm_layer.get();
    }
    //ssd_pruning
    else if (!strcmp(layerName, "conv13_mbox_loc_perm"))
    {
        assert(mConv13_mbox_loc_perm_layer.get() == nullptr);
        mConv13_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return mConv13_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv13_mbox_conf_perm"))
    {
        assert(mConv13_mbox_conf_perm_layer.get() == nullptr);
        mConv13_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return mConv13_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv14_2_mbox_loc_perm"))
    {
        assert(mConv14_2_mbox_loc_perm_layer.get() == nullptr);
        mConv14_2_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return mConv14_2_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv14_2_mbox_conf_perm"))
    {
        assert(mConv14_2_mbox_conf_perm_layer.get() == nullptr);
        mConv14_2_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return mConv14_2_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv15_2_mbox_loc_perm"))
    {
        assert(mConv15_2_mbox_loc_perm_layer.get() == nullptr);
        mConv15_2_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return mConv15_2_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv15_2_mbox_conf_perm"))
    {
        assert(mConv15_2_mbox_conf_perm_layer.get() == nullptr);
        mConv15_2_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return mConv15_2_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv16_2_mbox_loc_perm"))
    {
        assert(mConv16_2_mbox_loc_perm_layer.get() == nullptr);
        mConv16_2_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return mConv16_2_mbox_loc_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv16_2_mbox_conf_perm"))
    {
        assert(mConv16_2_mbox_conf_perm_layer.get() == nullptr);
        mConv16_2_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return mConv16_2_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv17_2_mbox_conf_perm"))
    {
        assert(mConv17_2_mbox_conf_perm_layer.get() == nullptr);
        mConv17_2_mbox_conf_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return mConv17_2_mbox_conf_perm_layer.get();
    }
    else if (!strcmp(layerName, "conv17_2_mbox_loc_perm"))
    {
        assert(mConv17_2_mbox_loc_perm_layer.get() == nullptr);
        mConv17_2_mbox_loc_perm_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPermutePlugin(serialData,serialLength), nvPluginDeleter);
        return mConv17_2_mbox_loc_perm_layer.get();
    }

    else if (!strcmp(layerName, "conv11_mbox_priorbox"))
    {
        assert(mConv11_mbox_priorbox_layer.get() == nullptr);

        mConv11_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(serialData,serialLength), nvPluginDeleter);
        return mConv11_mbox_priorbox_layer.get();
    }
    else if (!strcmp(layerName, "conv13_mbox_priorbox"))
    {
        assert(mConv13_mbox_priorbox_layer.get() == nullptr);
        mConv13_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(serialData,serialLength), nvPluginDeleter);
        return mConv13_mbox_priorbox_layer.get();
    }
    else if (!strcmp(layerName, "conv14_2_mbox_priorbox"))
    {
        assert(mConv14_2_mbox_priorbox_layer.get() == nullptr);
        mConv14_2_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(serialData,serialLength), nvPluginDeleter);
        return mConv14_2_mbox_priorbox_layer.get();
    }
    else if (!strcmp(layerName, "conv15_2_mbox_priorbox"))
    {
        assert(mConv15_2_mbox_priorbox_layer.get() == nullptr);
        mConv15_2_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(serialData,serialLength), nvPluginDeleter);
        return mConv15_2_mbox_priorbox_layer.get();
    }
    else if (!strcmp(layerName, "conv16_2_mbox_priorbox"))
    {
        assert(mConv16_2_mbox_priorbox_layer.get() == nullptr);
        mConv16_2_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(serialData,serialLength), nvPluginDeleter);
        return mConv16_2_mbox_priorbox_layer.get();
    }
    else if (!strcmp(layerName, "conv17_2_mbox_priorbox"))
    {
        assert(mConv17_2_mbox_priorbox_layer.get() == nullptr);
        mConv17_2_mbox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createSSDPriorBoxPlugin(serialData,serialLength), nvPluginDeleter);
        return mConv17_2_mbox_priorbox_layer.get();
    }

    else if (!strcmp(layerName, "mbox_loc"))
    {
        assert(mBox_loc_layer.get() == nullptr);
        mBox_loc_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData,serialLength), nvPluginDeleter);
        return mBox_loc_layer.get();
    }
    else if (!strcmp(layerName, "mbox_conf"))
    {
        assert(mBox_conf_layer.get() == nullptr);
        mBox_conf_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData,serialLength), nvPluginDeleter);
        return mBox_conf_layer.get();
    }
    else if (!strcmp(layerName, "mbox_priorbox"))
    {
        assert(mBox_priorbox_layer.get() == nullptr);
        mBox_priorbox_layer = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
                (createConcatPlugin(serialData,serialLength), nvPluginDeleter);
        return mBox_priorbox_layer.get();
    }
        //flatten
    else if (!strcmp(layerName, "conv11_mbox_conf_flat"))
    {
        assert(mConv11_mbox_conf_flat_layer.get() == nullptr);
        mConv11_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return mConv11_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv13_mbox_conf_flat"))
    {
        assert(mConv13_mbox_conf_flat_layer.get() == nullptr);
        mConv13_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return mConv13_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv14_2_mbox_conf_flat"))
    {
        assert(mConv14_2_mbox_conf_flat_layer.get() == nullptr);
        mConv14_2_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return mConv14_2_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv15_2_mbox_conf_flat"))
    {
        assert(mConv15_2_mbox_conf_flat_layer.get() == nullptr);
        mConv15_2_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return mConv15_2_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv16_2_mbox_conf_flat"))
    {
        assert(mConv16_2_mbox_conf_flat_layer.get() == nullptr);
        mConv16_2_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return mConv16_2_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv17_2_mbox_conf_flat"))
    {
        assert(mConv17_2_mbox_conf_flat_layer.get() == nullptr);
        mConv17_2_mbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return mConv17_2_mbox_conf_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv11_mbox_loc_flat"))
    {
        assert(mConv11_mbox_loc_flat_layer.get() == nullptr);
        mConv11_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return mConv11_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv13_mbox_loc_flat"))
    {
        assert(mConv13_mbox_loc_flat_layer.get() == nullptr);
        mConv13_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return mConv13_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv14_2_mbox_loc_flat"))
    {
        assert(mConv14_2_mbox_loc_flat_layer.get() == nullptr);
        mConv14_2_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return mConv14_2_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv15_2_mbox_loc_flat"))
    {
        assert(mConv15_2_mbox_loc_flat_layer.get() == nullptr);
        mConv15_2_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return mConv15_2_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv16_2_mbox_loc_flat"))
    {
        assert(mConv16_2_mbox_loc_flat_layer.get() == nullptr);
        mConv16_2_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return mConv16_2_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "conv17_2_mbox_loc_flat"))
    {
        assert(mConv17_2_mbox_loc_flat_layer.get() == nullptr);
        mConv17_2_mbox_loc_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return mConv17_2_mbox_loc_flat_layer.get();
    }
    else if (!strcmp(layerName, "mbox_conf_flatten"))
    {
        assert(mMbox_conf_flat_layer.get() == nullptr);
        mMbox_conf_flat_layer = std::unique_ptr<FlattenLayer>(new FlattenLayer(serialData,serialLength));
        return mMbox_conf_flat_layer.get();
    }
    //reshape
    else if (!strcmp(layerName, "mbox_conf_reshape"))
    {
        assert(mMbox_conf_reshape == nullptr);
        //num of class,by lcg
        mMbox_conf_reshape = std::unique_ptr<Reshape<5>>(new Reshape<5>(serialData, serialLength));
        return mMbox_conf_reshape.get();
    }
    //softmax
    else if (!strcmp(layerName, "mbox_conf_softmax"))
    {
        std::cout << "2_softmax" << std::endl;
        assert(mPluginSoftmax == nullptr);
        mPluginSoftmax = std::unique_ptr<SoftmaxPlugin>(new SoftmaxPlugin(serialData, serialLength));
        return mPluginSoftmax.get();
    }

    else if (!strcmp(layerName, "detection_out"))
    {
        std::cout << "2_detection_out" << std::endl;
        assert(mDetection_out.get() == nullptr);
        mDetection_out = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createSSDDetectionOutputPlugin(serialData, serialLength), nvPluginDeleter);
        return mDetection_out.get();
    }
    else
    {
        std::cout << "else" << std::endl;
        assert(0);
        return nullptr;
    }
}

bool PluginFactory::isPlugin(const char* name)
{
    return (!strcmp(name, "conv11_mbox_loc_perm")
            || !strcmp(name, "conv11_mbox_loc_flat")
            || !strcmp(name, "conv11_mbox_conf_perm")
            || !strcmp(name, "conv11_mbox_conf_flat")
            || !strcmp(name, "conv11_mbox_priorbox")
            || !strcmp(name, "conv13_mbox_loc_perm")
            || !strcmp(name, "conv13_mbox_loc_flat")
            || !strcmp(name, "conv13_mbox_conf_perm")
            || !strcmp(name, "conv13_mbox_conf_flat")
            || !strcmp(name, "conv13_mbox_priorbox")
            || !strcmp(name, "conv14_2_mbox_loc_perm")
            || !strcmp(name, "conv14_2_mbox_loc_flat")
            || !strcmp(name, "conv14_2_mbox_conf_perm")
            || !strcmp(name, "conv14_2_mbox_conf_flat")
            || !strcmp(name, "conv14_2_mbox_priorbox")
            || !strcmp(name, "conv15_2_mbox_loc_perm")
            || !strcmp(name, "conv15_2_mbox_loc_flat")
            || !strcmp(name, "conv15_2_mbox_conf_perm")
            || !strcmp(name, "conv15_2_mbox_conf_flat")
            || !strcmp(name, "conv15_2_mbox_priorbox")
            || !strcmp(name, "conv16_2_mbox_loc_perm")
            || !strcmp(name, "conv16_2_mbox_loc_flat")
            || !strcmp(name, "conv16_2_mbox_conf_perm")
            || !strcmp(name, "conv16_2_mbox_conf_flat")
            || !strcmp(name, "conv16_2_mbox_priorbox")
            || !strcmp(name, "conv17_2_mbox_loc_perm")
            || !strcmp(name, "conv17_2_mbox_loc_flat")
            || !strcmp(name, "conv17_2_mbox_conf_perm")
            || !strcmp(name, "conv17_2_mbox_conf_flat")
            || !strcmp(name, "conv17_2_mbox_priorbox")
            || !strcmp(name, "mbox_conf_reshape")
            || !strcmp(name, "mbox_conf_flatten")
            || !strcmp(name, "mbox_loc")
            || !strcmp(name, "mbox_conf")
            || !strcmp(name, "mbox_priorbox")
            || !strcmp(name, "mbox_conf_softmax")
            || !strcmp(name, "detection_out"));
}

void PluginFactory::destroyPlugin()
{
    std::cout << "distroyPlugin" << std::endl;
    //mNormalizeLayer.release();
    //mNormalizeLayer = nullptr;

    mConv11_mbox_conf_perm_layer.release();
    mConv11_mbox_conf_perm_layer = nullptr;
    mConv11_mbox_loc_perm_layer.release();
    mConv11_mbox_loc_perm_layer = nullptr;
    mConv13_mbox_conf_perm_layer.release();
    mConv13_mbox_conf_perm_layer = nullptr;
    mConv13_mbox_loc_perm_layer.release();
    mConv13_mbox_loc_perm_layer = nullptr;
    mConv14_2_mbox_conf_perm_layer.release();
    mConv14_2_mbox_conf_perm_layer = nullptr;
    mConv14_2_mbox_loc_perm_layer.release();
    mConv14_2_mbox_loc_perm_layer = nullptr;
    mConv15_2_mbox_conf_perm_layer.release();
    mConv15_2_mbox_conf_perm_layer = nullptr;
    mConv16_2_mbox_conf_perm_layer.release();
    mConv16_2_mbox_conf_perm_layer = nullptr;
    mConv16_2_mbox_loc_perm_layer.release();
    mConv16_2_mbox_loc_perm_layer = nullptr;
    mConv17_2_mbox_conf_perm_layer.release();
    mConv17_2_mbox_conf_perm_layer = nullptr;
    mConv17_2_mbox_loc_perm_layer.release();
    mConv17_2_mbox_loc_perm_layer = nullptr;
    mConv15_2_mbox_loc_perm_layer.release();
    mConv15_2_mbox_loc_perm_layer = nullptr;

    mConv13_mbox_priorbox_layer.release();
    mConv13_mbox_priorbox_layer = nullptr;
    mConv14_2_mbox_priorbox_layer.release();
    mConv14_2_mbox_priorbox_layer = nullptr;
    mConv15_2_mbox_priorbox_layer.release();
    mConv15_2_mbox_priorbox_layer = nullptr;
    mConv16_2_mbox_priorbox_layer.release();
    mConv16_2_mbox_priorbox_layer = nullptr;
    mConv11_mbox_priorbox_layer.release();
    mConv11_mbox_priorbox_layer = nullptr;
    mConv17_2_mbox_priorbox_layer.release();
    mConv17_2_mbox_priorbox_layer = nullptr;

    mBox_loc_layer.release();
    mBox_loc_layer = nullptr;
    mBox_conf_layer.release();
    mBox_conf_layer = nullptr;
    mBox_priorbox_layer.release();
    mBox_priorbox_layer = nullptr;

    mConv11_mbox_conf_flat_layer.release();
    mConv11_mbox_conf_flat_layer = nullptr;
    mConv13_mbox_conf_flat_layer.release();
    mConv13_mbox_conf_flat_layer = nullptr;
    mConv14_2_mbox_conf_flat_layer.release();
    mConv14_2_mbox_conf_flat_layer = nullptr;
    mConv15_2_mbox_conf_flat_layer.release();
    mConv15_2_mbox_conf_flat_layer = nullptr;
    mConv16_2_mbox_conf_flat_layer.release();
    mConv16_2_mbox_conf_flat_layer = nullptr;
    mConv17_2_mbox_conf_flat_layer.release();
    mConv17_2_mbox_conf_flat_layer = nullptr;
    mConv11_mbox_loc_flat_layer.release();
    mConv11_mbox_loc_flat_layer = nullptr;
    mConv13_mbox_loc_flat_layer.release();
    mConv13_mbox_loc_flat_layer = nullptr;
    mConv14_2_mbox_loc_flat_layer.release();
    mConv14_2_mbox_loc_flat_layer = nullptr;
    mConv15_2_mbox_loc_flat_layer.release();
    mConv15_2_mbox_loc_flat_layer = nullptr;
    mConv16_2_mbox_loc_flat_layer.release();
    mConv16_2_mbox_loc_flat_layer = nullptr;
    mConv17_2_mbox_loc_flat_layer.release();
    mConv17_2_mbox_loc_flat_layer = nullptr;
    mMbox_conf_flat_layer.release();
    mMbox_conf_flat_layer = nullptr;

    mMbox_conf_reshape.release();
    mMbox_conf_reshape = nullptr;
    mPluginSoftmax.release();
    mPluginSoftmax = nullptr;
    mDetection_out.release();
    mDetection_out = nullptr;

}
