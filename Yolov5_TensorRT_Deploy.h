#ifndef YOLOV5_TENSORRT_DEPLOY_H
#define YOLOV5_TENSORRT_DEPLOY_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common_api.h"


class Yolov5_TensorRT_Deploy : public ModelProcessor
{
public:
    Yolov5_TensorRT_Deploy(modelConfInfo_ info);
    ~Yolov5_TensorRT_Deploy();
    void get_model_info();
    cv::Mat pre_image_process(cv::Mat &image);
    void run_model(cv::Mat &input_image);
//    void post_image_process(std::vector<Ort::Value> &outputs, cv::Mat &inputimage);
    void post_image_process(cv::Mat &inputimage);
    void process();
    // show
    void set_Show_image(Show *imageShower);

    virtual void modelRunner();
    virtual void modelStop();

private:
    std::string model_path;
    std::string image_path;
    std::string label_path;
    std::string model;
    std::vector<std::string> labels;


    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;


    int input_h;
    int input_w;
    //cv::Mat input_image;
    int64_t inputSize = 1;
    int64_t outputSize = 1;

    int out_ch;
    int out_num;


    Show *image_show;

    float x_factor;
    float y_factor;

    int64 start_time;
    int64 end_time;

    cudaStream_t stream;

    Logger m_loger;

    void* buffers[2] = {nullptr, nullptr};
    std::vector<float> prob;

    IBuilder *m_builder =  nullptr;
    IRuntime *m_runtime = nullptr;
    ICudaEngine *m_cudaEngine = nullptr;
    nvinfer1::IExecutionContext *m_context = nullptr;
    char* trtModeStream = nullptr;

    bool m_runingFlag = true;
};

#endif // YOLOV5_ONNX_DEPLOY_H
