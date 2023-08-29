#ifndef YOLOV8_KEYPO_H
#define YOLOV8_KEYPO_H


#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <fstream>
#include "common_api.h"


class Yolov8_KeyPoint : public ModelProcessor
{
public:
    Yolov8_KeyPoint(std::string modelPath, std::string imagePath, std::string label_text, std::string modelType);
    ~Yolov8_KeyPoint();
    void get_model_info();
    cv::Mat pre_image_process(cv::Mat &image);
    void run_model(cv::Mat &input_image);
    void post_image_process(std::vector<Ort::Value> &outputs, cv::Mat &inputimage);
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
    Ort::SessionOptions session_options;
    Ort::Env env;
    std::wstring w_model_path;

    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;
    Ort::AllocatorWithDefaultOptions allocator;
    //Ort::Session session_;

    int input_h;
    int input_w;
    //cv::Mat input_image;

    int out_ch;
    int out_num;

    Ort::Session *session_;
    std::vector<Ort::Value> ort_outputs;

    Show *image_show;

    float x_factor;
    float y_factor;

    int64 start_time;
};

#endif // YOLOV8_KEYPO_H
