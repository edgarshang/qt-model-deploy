#include "Yolov5_Onnx_Deploy.h"
#include <QDebug>



Yolov5_Onnx_Deploy::Yolov5_Onnx_Deploy(std::string modelPath, std::string imagePath, std::string label_text)
{
    model_path = modelPath;
    image_path = imagePath;
    label_path = label_text;
}

Yolov5_Onnx_Deploy::~Yolov5_Onnx_Deploy()
{

}

void Yolov5_Onnx_Deploy::get_model_info()
{

}
cv::Mat Yolov5_Onnx_Deploy::pre_image_process(cv::Mat &image)
{
//    cv::Mat image;
    return image;
}
void Yolov5_Onnx_Deploy::run_model(cv::Mat &input_image)
{

}
void Yolov5_Onnx_Deploy::post_image_process(std::vector<Ort::Value> &outputs, cv::Mat &inputimage)
{

}
void Yolov5_Onnx_Deploy::process()
{

}
// show
void Yolov5_Onnx_Deploy::set_Show_image(Show *imageShower)
{

}

void Yolov5_Onnx_Deploy::modelRunner()
{
    qDebug() << "modelRunner()";
}
