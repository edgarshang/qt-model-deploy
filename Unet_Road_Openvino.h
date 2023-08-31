#ifndef UNET_ROAD_OPENVINO_H
#define UNET_ROAD_OPENVINO_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <fstream>
#include "common_api.h"


class Unet_Road_Openvino : public ModelProcessor
{

public:
    Unet_Road_Openvino(std::string modelPath, std::string imagePath, std::string label_text, std::string modelType);
    ~Unet_Road_Openvino();
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

#endif // UNET_ROAD_OPENVINO_H
