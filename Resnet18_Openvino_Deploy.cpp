#include "Resnet18_Openvino_Deploy.h"
#include <QDebug>

Resnet18_Openvino_Deploy::Resnet18_Openvino_Deploy(modelConfInfo_ info)
{
    model_path = info.modelPath;
    image_path = info.imagePath;
    label_path = info.label_text;
    model = info.modelType;
    scoresThr = info.scoresThreshold;
    confindenceThr = info.confienceThreshold;
//    configureInfo = info;
}

Resnet18_Openvino_Deploy::~Resnet18_Openvino_Deploy()
{



}

void Resnet18_Openvino_Deploy::get_model_info()
{
    compiled_model = ie.compile_model(model_path, "CPU");
    infer_request = compiled_model.create_infer_request();

    input_tensor = infer_request.get_input_tensor();
    ov::Shape tensor_shape =  input_tensor.get_shape();
    qDebug() << tensor_shape;
    int num_channels = tensor_shape[1];
    input_h = tensor_shape[2];
    input_w = tensor_shape[3];
}

cv::Mat Resnet18_Openvino_Deploy::pre_image_process(cv::Mat &image)
{
    start_time = cv::getTickCount();
    // set input image
    cv::Mat rgb, blob;
    // RGB order
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, blob, cv::Size(input_w, input_h));
    blob.convertTo(blob, CV_32F);
    blob = blob / 255.0;
    cv::subtract(blob, cv::Scalar(0.485, 0.456, 0.406), blob);
    cv::divide(blob, cv::Scalar(0.229, 0.224, 0.225), blob);

    return blob;
}

void Resnet18_Openvino_Deploy::run_model(cv::Mat &input_image)
{
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

void Resnet18_Openvino_Deploy::post_image_process(cv::Mat &inputimage)
{
   auto output_tensor = infer_request.get_output_tensor();
   out_num = output_tensor.get_shape()[1];
   out_ch =output_tensor.get_shape()[2];

   // 1x 1000
   const float* pdata = (float*)output_tensor.data();
   cv::Mat prob(out_num, out_ch, CV_32F, (float*)pdata);

   cv::Point maxL, minL;
   double maxv, minv;
   cv::minMaxLoc(prob, &minv, &maxv, &minL, &maxL);
   int max_index = maxL.x;
//    std::cout << "label id: " << max_index << std::endl;
   cv::putText(inputimage, labels[max_index], cv::Point(50, 80), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2, 8);

   float t = (cv::getTickCount() - start_time) / static_cast<float>(cv::getTickFrequency());
   cv::putText(inputimage, cv::format("FPS: %.2f", 1.0/t), cv::Point(20,30), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);

//    image_show->imageshow(inputimage);

}

void Resnet18_Openvino_Deploy::process()
{
    labels = Common_API::readClassNames(label_path);
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
void Resnet18_Openvino_Deploy::set_Show_image(Show *imageShower)
{
    image_show = imageShower;
}

void Resnet18_Openvino_Deploy::modelRunner()
{
    this->process();
}
