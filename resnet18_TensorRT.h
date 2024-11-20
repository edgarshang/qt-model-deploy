#ifndef RESNET18_TENSORRT_H
#define RESNET18_TENSORRT_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common_api.h"

using namespace nvinfer1;
using namespace nvonnxparser;

class resnet18_TensorRT : public ModelProcessor
{
public:
    resnet18_TensorRT(modelConfInfo_ info);
    ~resnet18_TensorRT();
    void get_model_info();
    cv::Mat pre_image_process(cv::Mat &image);
    void run_model(cv::Mat &input_image);
    void post_image_process(cv::Mat &inputimage);
    void process();
    // show
    void set_Show_image(Show *imageShower);

    virtual void modelRunner();

private:
    std::string model_path;
    std::string image_path;
    std::string label_path;
    std::string model;
    std::vector<std::string> labels;
    Logger m_loger;
    float scoresThr;
    float confindenceThr;

    void* buffers[2] = {nullptr, nullptr};
    std::vector<float> prob;

    IBuilder *builder =  nullptr;
    IRuntime *m_runtime = nullptr;
    ICudaEngine *m_cudaEngine = nullptr;
    nvinfer1::IExecutionContext *m_context = nullptr;

    int m_inputW = 224;
    int m_inputH = 224;
    int m_outputSize = 1000;
    Show *image_show = nullptr;
    char* trtModeStream = nullptr;

//    int64 start_time;
//    int64 end_time;
};

#endif // RESNET18_TENSORRT_H
