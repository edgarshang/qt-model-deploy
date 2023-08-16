#ifndef ORT_TUTORIAL_H
#define ORT_TUTORIAL_H

#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <fstream>
#include "common_api.h"

class ort_tutorial : public ModelProcessor
{
public:
    ort_tutorial(std::string modelPath, std::string imagePath, std::string label_text);
    ~ort_tutorial();
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
};

#endif // ORT_TUTORIAL_H
