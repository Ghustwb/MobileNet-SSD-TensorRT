#include "pluginImplement.h"
#include "mathFunctions.h"
#include <vector>
#include <algorithm>

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
        params.codeType = CodeTypeSSD::CENTER_SIZE;
        params.keepTopK = 100;
        params.shareLocation = true;
        params.varianceEncodedInTarget = false;
        params.topK = 100;
        params.nmsThreshold = 0.45;
        params.numClasses = 5;
        params.inputOrder[0] = 0;
        params.inputOrder[1] = 1;
        params.inputOrder[2] = 2;
        params.confidenceThreshold = 0.6;
	params.confSigmoid = true;
	params.isNormalized = true;

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
