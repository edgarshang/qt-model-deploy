#include "DeepLabV3_Openvino_Deploy.h"
#include <QDebug>

DeepLabV3_Openvino_Deploy::DeepLabV3_Openvino_Deploy(modelConfInfo_ info)
{
    model_path = info.modelPath;
    image_path = info.imagePath;
    label_path = info.label_text;
    model = info.modelType;
    scoresThr = info.scoresThreshold;
    confindenceThr = info.confienceThreshold;
}

DeepLabV3_Openvino_Deploy::~DeepLabV3_Openvino_Deploy()
{



}

void DeepLabV3_Openvino_Deploy::get_model_info()
{
    compiled_model = ie.compile_model(model_path, "CPU");
    infer_request = compiled_model.create_infer_request();

    input_tensor = infer_request.get_input_tensor();
}

cv::Mat DeepLabV3_Openvino_Deploy::pre_image_process(cv::Mat &image)
{
    start_time = cv::getTickCount();
    input_h = image.rows;
    input_w = image.cols;

    cv::Mat gblob;
    image.convertTo(gblob, CV_32F);
    gblob = gblob / 255.0;

    return gblob;
}

void DeepLabV3_Openvino_Deploy::run_model(cv::Mat &input_image)
{
    size_t batch = 1;
    size_t channal = 3;
    ov::Shape input_shape = {batch, channal, input_h, input_w};
    input_tensor.set_shape(input_shape);

    qDebug() << input_tensor.get_shape();

    float* image_data = input_tensor.data<float>();

   for (size_t c = 0; c < 3; c++) {
       for (size_t h = 0; h < input_h; h++) {
           for (size_t w = 0; w < input_w; w++) {
               size_t index = c * input_w * input_h + h * input_w + w;
               image_data[index] = input_image.at<cv::Vec3f>(h, w)[c];
           }
       }
   }

       infer_request.infer();
}

void DeepLabV3_Openvino_Deploy::post_image_process(cv::Mat &inputimage)
{
   auto output_tensor_0 = infer_request.get_output_tensor(0);

   std::vector<cv::Vec3b> color_table;
   color_table.push_back(cv::Vec3b(0, 0, 0));
   cv::RNG rng(cv::getTickCount());
   const float* mask_data = (float*)output_tensor_0.data();

   ov::Shape outShape = output_tensor_0.get_shape();
   int num_cn = outShape[1];
   int out_h = outShape[2];
   int out_w = outShape[3];
   qDebug() <<num_cn<< "x" << out_h << "x" << out_w;

   for (int i = 1; i < num_cn; i++) {
       color_table.push_back(cv::Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
   }

   int step = out_h * out_w;
   cv::Mat result = cv::Mat::zeros(cv::Size(out_w, out_h), CV_8UC3);
   for (int row = 0; row < out_h; row++) {
       for (int col = 0; col < out_w; col++) {
           int max_index = 0;
           float max_prob = mask_data[row*out_w + col];
           for (int cn = 1; cn < num_cn; cn++) {
               float prob = mask_data[cn*step + row*out_w + col];
               if (prob > max_prob) {
                   max_prob = prob;
                   max_index = cn;
               }
           }
           result.at<cv::Vec3b>(row, col) = color_table[max_index];
       }
   }

   cv::addWeighted(inputimage, 0.7, result, 0.3, 0, inputimage);
   cv::Mat gray;
   cv::cvtColor(result, gray, cv::COLOR_BGR2GRAY);
   cv::Mat dst;
//    cv::bitwise_and(inputimage, inputimage, dst, gray);

   // compute the fps
   float t = (cv::getTickCount() - start_time) / static_cast<float>(cv::getTickFrequency());
   cv::putText(inputimage, cv::format("FPS: %.2f", 1.0/t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255,0,0), 2, 8);
}

void DeepLabV3_Openvino_Deploy::process()
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
void DeepLabV3_Openvino_Deploy::set_Show_image(Show *imageShower)
{
    image_show = imageShower;
}

void DeepLabV3_Openvino_Deploy::modelRunner()
{
    this->process();
}
