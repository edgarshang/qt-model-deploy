#include "Unet_Road_Openvino.h"
#include <QDebug>

Unet_Road_Openvino::Unet_Road_Openvino(std::string modelPath, std::string imagePath, std::string label_text, std::string modelType)
{
    model_path = modelPath;
    image_path = imagePath;
    label_path = label_text;
    model = modelType;
}

Unet_Road_Openvino::~Unet_Road_Openvino()
{



}

void Unet_Road_Openvino::get_model_info()
{
    compiled_model = ie.compile_model("D:/project/ort-deploy/unet_road.onnx", "CPU");
    infer_request = compiled_model.create_infer_request();

    input_tensor = infer_request.get_input_tensor();
    ov::Shape tensor_shape =  input_tensor.get_shape();
    qDebug() << tensor_shape;
    int num_channels = tensor_shape[1];
    input_h = tensor_shape[2];
    input_w = tensor_shape[3];
}

cv::Mat Unet_Road_Openvino::pre_image_process(cv::Mat &image)
{
    start_time = cv::getTickCount();
    cv::Mat gray, gblob;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, gblob, cv::Size(input_w, input_h));
    gblob.convertTo(gblob, CV_32F);
    gblob = gblob / 255.0;

    return gblob;
}

void Unet_Road_Openvino::run_model(cv::Mat &input_image)
{
       float* image_data = input_tensor.data<float>();

       for(int row = 0; row < input_h; row++)
       {
           for(int col = 0; col < input_w; col++)
           {
               image_data[row*input_w +col] = input_image.at<float>(row, col);
           }
       }

       infer_request.infer();
}

void Unet_Road_Openvino::post_image_process(cv::Mat &inputimage)
{
       auto output_tensor = infer_request.get_output_tensor();
       const float* detection = (float*)output_tensor.data();
       ov::Shape out_shape = output_tensor.get_shape();
//       const int out_c = out_shape[1];
       const int out_h = out_shape[2];
       const int out_w = out_shape[3];
       cv::Mat result = cv::Mat::zeros(cv::Size(out_w, out_h), CV_8UC1);

       // 解析结果
       for(int row = 0; row < out_h; row++)
       {
           for(int col = 0; col < out_w; col++)
           {
               float c1 = detection[row*out_w + col];
               float c2 = detection[out_h* out_w + row*out_w + col];
               if(c1 > c2)
               {
                   result.at<uchar>(row, col) = 0;
               }else
               {
                   result.at<uchar>(row, col) = 255;
               }
           }
       }

       cv::Mat mask, binary;
       cv::resize(result, mask, cv::Size(inputimage.cols, inputimage.rows));
       cv::threshold(mask, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
       std::vector<std::vector<cv::Point>> contours;
       cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
       cv::drawContours(inputimage, contours, -1, cv::Scalar(0,0,255), -1, 8);

       // compute the fps
       float t = (cv::getTickCount() - start_time) / static_cast<float>(cv::getTickFrequency());
       cv::putText(inputimage, cv::format("FPS: %.2f", 1.0/t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255,0,0), 2, 8);
}

void Unet_Road_Openvino::process()
{
    this->get_model_info();

    QString path = QString::fromStdString(image_path);

    if(path.endsWith(".mp4") || path.endsWith(".avi"))
    {
        cv::VideoCapture capture(path.toStdString());
        if(capture.isOpened())
        {
            cv::Mat frame;
            while(true)
            {
                bool ret = capture.read(frame);
                if(!ret)
                {
                    break;
                }

                cv::Mat model_input = this->pre_image_process(frame);
                this->run_model(model_input);
                this->post_image_process(frame);
                image_show->imageshow(frame);
            }

            capture.release();
        }
    }
    else{
        cv::Mat image = cv::imread(path.toStdString());
        cv::Mat model_input = this->pre_image_process(image);
        this->run_model(model_input);
        this->post_image_process(image);
        image_show->imageshow(image);
    }
}

// show
void Unet_Road_Openvino::set_Show_image(Show *imageShower)
{
    image_show = imageShower;
}

void Unet_Road_Openvino::modelRunner()
{
    this->process();
}

