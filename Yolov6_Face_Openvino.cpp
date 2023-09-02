#include "Yolov6_Face_Openvino.h"
#include <QDebug>

Yolov6_Face_Openvino_Deploy::Yolov6_Face_Openvino_Deploy(modelConfInfo_ info)
{
    model_path = info.modelPath;
    image_path = info.imagePath;
    label_path = info.label_text;
    model = info.modelType;
    scoresThr = info.scoresThreshold;
    confindenceThr = info.confienceThreshold;
//    configureInfo = info;
}

Yolov6_Face_Openvino_Deploy::~Yolov6_Face_Openvino_Deploy()
{



}

void Yolov6_Face_Openvino_Deploy::get_model_info()
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

cv::Mat Yolov6_Face_Openvino_Deploy::pre_image_process(cv::Mat &image)
{
    start_time = cv::getTickCount();
    int w = image.cols;
    int h = image.rows;

    int _max = std::max(h,w);

    cv::Mat image_m = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
    cv::Rect roi(0,0,w,h);
    image.copyTo(image_m(roi));
    x_factor = image_m.cols / static_cast<float>(640);
    y_factor = image_m.rows / static_cast<float>(640);
    cv::Mat gblob;
    cv::resize(image_m, gblob, cv::Size(input_w, input_h));
    gblob.convertTo(gblob, CV_32F);
    gblob = gblob / 255.0;

    m1_factor = cv::Mat::zeros(cv::Size(2, 5), CV_32FC1);
    for(int i = 0; i< 5; i++)
    {
        m1_factor.at<float>(i,0) = x_factor;
        m1_factor.at<float>(i,1) = y_factor;
    }


    return gblob;
}

void Yolov6_Face_Openvino_Deploy::run_model(cv::Mat &input_image)
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

void Yolov6_Face_Openvino_Deploy::post_image_process(cv::Mat &inputimage)
{
   auto output_tensor = infer_request.get_output_tensor();
   const float* pdata = (float*)output_tensor.data();
   out_num = output_tensor.get_shape()[1];
   out_ch =output_tensor.get_shape()[2];

   // 后处理 1x8400x16 16 = box(4) + landmark(10) + sorces(1) + confinence(1)
   std::vector<cv::Rect> boxes;
   std::vector<float> confidences;
   std::vector<cv::Mat> keypoints;
   cv::Mat det_output(out_num, out_ch, CV_32F, (float*)pdata);

   // 16 = xyxy(4) + landmark(10) + socres(1) + conf(1)
   for(int i = 0; i < det_output.rows; i++)
   {

       float conf = det_output.at<float>(i,15);
//        qDebug() << "conf = " << conf;
       if(conf < 0.7)
       {
           continue;
       }


       float cx = det_output.at<float>(i,0);
       float cy = det_output.at<float>(i,1);
       float ow = det_output.at<float>(i,2);
       float oh = det_output.at<float>(i,3);

       int x = static_cast<int>((cx - 0.5*ow) * x_factor);
       int y = static_cast<int>((cy - 0.5*oh) * y_factor);
       int width = static_cast<int>(ow*x_factor);
       int height = static_cast<int>(oh*y_factor);

       cv::Rect box;
       box.x = x;
       box.y = y;
       box.width = width;
       box.height = height;

       boxes.push_back(box);
       confidences.push_back(conf);
       cv::Mat pts = det_output.row(i).colRange(4, 14);
       keypoints.push_back(pts);
   }

   // NMS
   std::vector<int> indexes;
   cv::dnn::NMSBoxes(boxes, confidences, (float)(0.25), (float)(0.45), indexes);
   for(size_t i = 0; i < indexes.size(); i++)
   {
       int idx = indexes[i];
       cv::rectangle(inputimage, boxes[idx], cv::Scalar(0,0,255), 2, 8,0);
       cv::putText(inputimage, cv::format("face %.2f", confidences[idx]) , boxes[idx].tl(),
                   cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0,255,0), 2, 8);
       cv::Mat keyPoint = keypoints[i];
       keyPoint = keyPoint.reshape(0,5);
       cv::Mat kp;

       cv::multiply(keyPoint, m1_factor, kp);
       kp = kp.reshape(0,10);

       const float* kpts_data = &kp.at<float>(0,0);
//        Common_API::draw_pose_keyPoint(kpts_data, inputimage);
       // render all key point circles
       for (int c = 0; c < 5; c++) {
           cv::circle(inputimage, cv::Point(kpts_data[c * 2], kpts_data[c * 2 + 1]), 4, cv::Scalar(0, 255, 0), 3, 8, 0);
       }
   }

   // compute the fps
   float t = (cv::getTickCount() - start_time) / static_cast<float>(cv::getTickFrequency());
   cv::putText(inputimage, cv::format("FPS: %.2f", 1.0/t), cv::Point(20,40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
}

void Yolov6_Face_Openvino_Deploy::process()
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
void Yolov6_Face_Openvino_Deploy::set_Show_image(Show *imageShower)
{
    image_show = imageShower;
}

void Yolov6_Face_Openvino_Deploy::modelRunner()
{
    this->process();
}
