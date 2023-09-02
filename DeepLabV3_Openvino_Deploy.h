#ifndef DEEPLABV3_OPENVINO_DEPLOY_H
#define DEEPLABV3_OPENVINO_DEPLOY_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <fstream>
#include "common_api.h"

class DeepLabV3_Openvino_Deploy : public ModelProcessor
{

public:
    DeepLabV3_Openvino_Deploy(modelConfInfo_ info);
    ~DeepLabV3_Openvino_Deploy();
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
    float scoresThr;
    float confindenceThr;

    size_t input_h;
    size_t input_w;

    Show *image_show;
    int64 start_time;
    ov::Core ie;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    ov::Tensor input_tensor;
};

#endif // DEEPLABV3_OPENVINO_DEPLOY_H
