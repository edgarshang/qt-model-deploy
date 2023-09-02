#ifndef RESNET18_OPENVINO_DEPLOY_H
#define RESNET18_OPENVINO_DEPLOY_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <fstream>
#include "common_api.h"


class Resnet18_Openvino_Deploy : public ModelProcessor
{

public:
    Resnet18_Openvino_Deploy(modelConfInfo_ info);
    ~Resnet18_Openvino_Deploy();
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
    float scoresThr;
    float confindenceThr;
//    modelConInfo_ configureInfo;


    int input_h;
    int input_w;

    int out_ch;
    int out_num;



    Show *image_show;

    float x_factor;
    float y_factor;

    int64 start_time;
    ov::Core ie;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    ov::Tensor input_tensor;
};

#endif // RESNET18_OPENVINO_DEPLOY_H
