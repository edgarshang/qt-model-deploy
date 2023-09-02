#include "keyPointRcnn_Openvino_Deploy.h"
#include <QDebug>

keyPointRcnn_Openvino_Deploy::keyPointRcnn_Openvino_Deploy(modelConfInfo_ info)
{
    model_path = info.modelPath;
    image_path = info.imagePath;
    label_path = info.label_text;
    model = info.modelType;
    scoresThr = info.scoresThreshold;
    confindenceThr = info.confienceThreshold;
}

keyPointRcnn_Openvino_Deploy::~keyPointRcnn_Openvino_Deploy()
{



}

void keyPointRcnn_Openvino_Deploy::get_model_info()
{
    compiled_model = ie.compile_model(model_path, "CPU");
    infer_request = compiled_model.create_infer_request();

    input_tensor = infer_request.get_input_tensor();
}

cv::Mat keyPointRcnn_Openvino_Deploy::pre_image_process(cv::Mat &image)
{
    start_time = cv::getTickCount();
    input_h = image.rows;
    input_w = image.cols;

    cv::Mat gblob;
    image.convertTo(gblob, CV_32F);
    gblob = gblob / 255.0;

    return gblob;
}

void keyPointRcnn_Openvino_Deploy::run_model(cv::Mat &input_image)
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

void keyPointRcnn_Openvino_Deploy::post_image_process(cv::Mat &inputimage)
{
   auto output_tensor_0 = infer_request.get_output_tensor(0);
   auto output_tensor_1 = infer_request.get_output_tensor(1);
   auto output_tensor_2 = infer_request.get_output_tensor(2);
   auto output_tensor_3 = infer_request.get_output_tensor(3);
   const float* boxes = (float*)output_tensor_0.data();
   const int64* labels  = (int64*)output_tensor_1.data();
   const float* scores = (float*)output_tensor_2.data();
   const float* multiple_kypts = (float*)output_tensor_3.data();

   ov::Shape outShape = output_tensor_0.get_shape();
   int rows = outShape[0];

   std::cout << "fixed number: " << rows << std::endl;

   cv::Mat det_output(rows, 4, CV_32F, (float*)boxes);
   for(int i = 0; i < det_output.rows; i++)
   {
       double conf = scores[i];
       int cid = labels[i] - 1;
       // 置信度在0-1之间
       if(conf > 0.85)
       {
           float x1 = det_output.at<float>(i,0);
           float y1 = det_output.at<float>(i,1);
           float x2 = det_output.at<float>(i,2);
           float y2 = det_output.at<float>(i,3);

           cv::Rect box;
           box.x = x1;
           box.y = y1;
           box.width = x2 - x1;
           box.height = y2 - y1;

           cv::rectangle(inputimage, box, cv::Scalar(0,0,255), 2, 8, 0);
           cv::putText(inputimage, cv::format("%s_%.2f", labels_name[cid].c_str(), conf), box.tl(), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 255, 0), 2, 8);

           // draw key points
           const float* kpts_data = &multiple_kypts[i * 51];
           Common_API::draw_pose_keyPoint(kpts_data, inputimage);
       }
   }

   // compute the fps
   float t = (cv::getTickCount() - start_time) / static_cast<float>(cv::getTickFrequency());
   cv::putText(inputimage, cv::format("FPS: %.2f", 1.0/t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255,0,0), 2, 8);
}

void keyPointRcnn_Openvino_Deploy::process()
{
    labels_name = Common_API::readClassNames(label_path);
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
void keyPointRcnn_Openvino_Deploy::set_Show_image(Show *imageShower)
{
    image_show = imageShower;
}

void keyPointRcnn_Openvino_Deploy::modelRunner()
{
    this->process();
}
