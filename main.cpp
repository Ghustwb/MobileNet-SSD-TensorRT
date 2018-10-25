#include "common.h"
#include "cudaUtility.h"
#include "mathFunctions.h"
#include "pluginImplement.h"
#include "tensorNet.h"
#include "loadImage.h"
#include <chrono>

#include <QList>
#include <QFile>
#include <QDir>
#include <QDebug>
#include <QTime>

//5类
const char* model  = "/home/lcg/Desktop/mobileNet/MobileNetSSD_deploy_iplugin.prototxt";
const char* weight = "/home/lcg/Desktop/mobileNet/MobileNetSSD_deploy_150000.caffemodel";

const char* INPUT_BLOB_NAME = "data";

const char* OUTPUT_BLOB_NAME = "detection_out";
static const uint32_t BATCH_SIZE = 1;


class Timer {
public:
    void tic() {
        start_ticking_ = true;
        start_ = std::chrono::high_resolution_clock::now();
    }
    void toc() {
        if (!start_ticking_)return;
        end_ = std::chrono::high_resolution_clock::now();
        start_ticking_ = false;
        t = std::chrono::duration<double, std::milli>(end_ - start_).count();
        //std::cout << "Time: " << t << " ms" << std::endl;
    }
    double t;
private:
    bool start_ticking_ = false;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_;
};


/* *
 * @TODO: unifiedMemory is used here under -> ( cudaMallocManaged )
 * */
float* allocateMemory(DimsCHW dims, char* info)
{
    float* ptr;
    size_t size;
    std::cout << "Allocate memory: " << info << std::endl;
    size = BATCH_SIZE * dims.c() * dims.h() * dims.w();
    assert(!cudaMallocManaged( &ptr, size*sizeof(float)));
    return ptr;
}


QList<QString> readDic(QString dirName)
{
    QList<QString> result;
    QDir dir(dirName);
    dir.setFilter(QDir::Files | QDir::Hidden | QDir::NoSymLinks);

    QFileInfoList list = dir.entryInfoList();
    for (int i = 0; i < list.size(); ++i) {
        QFileInfo fileInfo = list.at(i);
        result.append(fileInfo.absoluteFilePath());
    }
    return result;
}

void displayBbox(cv::Mat image,std::vector<std::vector<float> > result,double msTime)
{
    QString qClassIndex;
    QString qConfidence;
    qDebug()<<"Time:  " << msTime;
    int fps = 1000 / msTime;
    QString qFPS = QString("%1").arg(fps);

    for(int i = 0;i < result.size();i++)
    {
        const vector<float>& d = result[i];
        const float classIndex = d[1];
        std::cout <<"class: " << classIndex << std::endl;
        const float score = d[2];//confidence
        qConfidence = QString("%1").arg(score);
        switch (int(classIndex)) {
        case 1:
            qClassIndex = "car";
            break;
        case 2:
            qClassIndex = "bus";
            break;
        case 3:
            qClassIndex = "person";
            break;
        case 4:
            qClassIndex = "truck";
            break;
        }

        int x1 = static_cast<int>(d[3] * image.cols);
        int y1 = static_cast<int>(d[4] * image.rows);
        int x2 = static_cast<int>(d[5] * image.cols);
        int y2 = static_cast<int>(d[6] * image.rows);
        cv::rectangle(image,cv::Rect2f(cv::Point(x1,y1),cv::Point(x2,y2)),cv::Scalar(255,0,255),1);
        cv::putText(image,qConfidence.toStdString(),cv::Point(x1,y1),1,1,cv::Scalar(255,0,255),1);
        cv::putText(image,qClassIndex.toStdString(),cv::Point(x1,y2),1,2,cv::Scalar(255,0,0),2);
        cv::putText(image,"FPS:" + qFPS.toStdString(),cv::Point(20,20),1,2,cv::Scalar(0,0,255),2);
    }
    cv::imshow("MobileNet-SSD",image);
    cv::waitKey(0);
}

void loadImg( cv::Mat &input, int re_width, int re_height, float *data_unifrom,const float3 mean,const float scale )
{
    int i;
    int j;
    int line_offset;
    int offset_g;
    int offset_r;
    cv::Mat dst;

    unsigned char *line = NULL;
    float *unifrom_data = data_unifrom;

    cv::resize( input, dst, cv::Size( re_width, re_height ), (0.0), (0.0), cv::INTER_LINEAR );
    offset_g = re_width * re_height;
    offset_r = re_width * re_height * 2;
    for( i = 0; i < re_height; ++i )
    {
        line = dst.ptr< unsigned char >( i );
        line_offset = i * re_width;
        for( j = 0; j < re_width; ++j )
        {
            // b
            unifrom_data[ line_offset + j  ] = (( float )(line[ j * 3 ] - mean.x) * scale);
            // g
            unifrom_data[ offset_g + line_offset + j ] = (( float )(line[ j * 3 + 1 ] - mean.y) * scale);
            // r
            unifrom_data[ offset_r + line_offset + j ] = (( float )(line[ j * 3 + 2 ] - mean.z) * scale);
        }
    }
}

int main(int argc, char *argv[])
{
    std::vector<std::string> output_vector = {OUTPUT_BLOB_NAME};
    TensorNet tensorNet;
    tensorNet.LoadNetwork(model,weight,INPUT_BLOB_NAME, output_vector,BATCH_SIZE);
    //tensorNet.caffeToTRTModel( model, weight, std::vector<std::string>{ output_vector }, BATCH_SIZE);
    //tensorNet.createInference();

    DimsCHW dimsData = tensorNet.getTensorDims(INPUT_BLOB_NAME);
    DimsCHW dimsOut  = tensorNet.getTensorDims(OUTPUT_BLOB_NAME);

    float* data    = allocateMemory( dimsData , (char*)"input blob");
    std::cout << "allocate data" << std::endl;
    float* output  = allocateMemory( dimsOut  , (char*)"output blob");
    std::cout << "allocate output" << std::endl;
    int height = 300;
    int width  = 300;

    cv::Mat frame,srcImg;

    void* imgCPU;
    void* imgCUDA;
    Timer timer;
    //QTime timer;

    QString imgPath = "/media/lcg/Disk/Data/testDataSet";
    QList<QString> imgPathList = readDic(imgPath);
    for(int i = 0;i < imgPathList.size();i++)
    {
        std::string imgFile = imgPathList.at(i).toStdString();

        frame = cv::imread(imgFile);
        srcImg = frame.clone();
        cv::resize(frame, frame, cv::Size(300,300));
        const size_t size = width * height * sizeof(float3);

        if( CUDA_FAILED( cudaMalloc( &imgCUDA, size)) )
        {
            cout <<"Cuda Memory allocation error occured."<<endl;
            return false;
        }

        void* imgData = malloc(size);
        memset(imgData,0,size);

        loadImg(frame,height,width,(float*)imgData,make_float3(127.5,127.5,127.5),0.007843);
        cudaMemcpyAsync(imgCUDA,imgData,size,cudaMemcpyHostToDevice);

        void* buffers[] = { imgCUDA, output };

        timer.tic();
        tensorNet.imageInference( buffers, output_vector.size() + 1, BATCH_SIZE);
        timer.toc();
        double msTime = timer.t;

        vector<vector<float> > detections;

        for (int k=0; k<100; k++)
        {
            if(output[7*k+1] == -1)
                break;
            float classIndex = output[7*k+1];
            float confidence = output[7*k+2];
            float xmin = output[7*k + 3];
            float ymin = output[7*k + 4];
            float xmax = output[7*k + 5];
            float ymax = output[7*k + 6];
            std::cout << xmin << " , " << ymin<< " , " << xmax<< " , " << ymax << std::endl;
            vector<float> detection = {0,classIndex,confidence,xmin,ymin,xmax,ymax};
            detections.push_back(detection);
        }
        displayBbox(srcImg,detections,msTime);
        free(imgData);
        cudaFree(imgCUDA);
        cudaFreeHost(imgCPU);
    }
    cudaFree(output);
    tensorNet.destroy();
    return 0;
}
