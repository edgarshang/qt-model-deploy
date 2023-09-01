#include "Yolov5_Seg_Openvino_Deploy.h"
#include <QDebug>

Yolov5_Seg_Openvino_Deploy::Yolov5_Seg_Openvino_Deploy(modelConfInfo_ info)
{
    model_path = info.modelPath;
    image_path = info.imagePath;
    label_path = info.label_text;
    model = info.modelType;
    scoresThr = info.scoresThreshold;
    confindenceThr = info.confienceThreshold;
//    configureInfo = info;
    sx = 160.0f / 640.0f;
    sy = 160.0f / 640.0f;
}

Yolov5_Seg_Openvino_Deploy::~Yolov5_Seg_Openvino_Deploy()
{



}

void Yolov5_Seg_Openvino_Deploy::get_model_info()
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

cv::Mat Yolov5_Seg_Openvino_Deploy::pre_image_process(cv::Mat &image)
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


    return gblob;
}

void Yolov5_Seg_Openvino_Deploy::run_model(cv::Mat &input_image)
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

void Yolov5_Seg_Openvino_Deploy::post_image_process(cv::Mat &inputimage)
{
   auto output_tensor_0 = infer_request.get_output_tensor(0);
   auto output_tensor_1 = infer_request.get_output_tensor(1);
   const float* pdata = (float*)output_tensor_0.data();
   const float* mdata = (float*)output_tensor_1.data();
   out_num = output_tensor_0.get_shape()[1];
   out_ch =output_tensor_0.get_shape()[2];

   for(int i = 0; i < output_tensor_0.get_shape().size(); i++)
   {
        qDebug() << output_tensor_0.get_shape()[i];
   }

   qDebug() << "===========";

   for(int i = 0; i < output_tensor_1.get_shape().size(); i++)
   {
        qDebug() << output_tensor_1.get_shape()[i];
   }

   // 后处理 1x25200x117(85+32) 85-box conf 80- min/max + 32 mask
   std::vector<cv::Rect> boxes;
   std::vector<int> classIds;
   std::vector<float> confidences;
   cv::Mat det_output(out_num, out_ch, CV_32F, (float*)pdata);
//   cv::Mat mask_output(32, 25600, CV_32F, (float*)mdata);
   std::vector<cv::Mat> masks;
   cv::Mat mask1(32, 25600, CV_32F, (float*)mdata);

   det_output = (model == YOLOV5_SEG ? det_output : det_output.t());

   for(int i = 0; i < det_output.rows; i++)
   {
       if (model == YOLOV5_SEG)
       {
           float conf = det_output.at<float>(i,4);
           if(conf < 0.45)
           {
               continue;
           }
       }


       cv::Mat classes_scores = det_output.row(i).colRange((model == YOLOV5_SEG ? 5 : 4), (model == YOLOV5_SEG ? (out_ch - 32) : (out_num - 32)));
       cv::Point classIdPoint;
       double score;
       cv::minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

       // 置信度0-1之间
       if( score > 0.25)
       {

           cv::Mat mask2 = det_output.row(i).colRange(model == YOLOV5_SEG ? (out_ch - 32): (out_num - 32), model == YOLOV5_SEG ? (out_ch) : (out_num));

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
           classIds.push_back(classIdPoint.x);
           confidences.push_back(score);
           masks.push_back(mask2);
       }
   }

   // NMS
   std::vector<int> indexes;
   cv::dnn::NMSBoxes(boxes, confidences, (float)(0.25), (float)(0.45), indexes);
   cv::Mat rgb_mask = cv::Mat::zeros(inputimage.size(), inputimage.type());

   for(size_t i = 0; i < indexes.size(); i++)
   {
       int idx = indexes[i];
       int cid = classIds[idx];

       cv::Rect box = boxes[idx];
       int x1 = std::max(0, box.x);
       int y1 = std::max(0, box.y);
       int x2 = std::max(0, box.br().x);
       int y2 = std::max(0, box.br().y);
       cv::Mat m2 = masks[idx];
       cv::Mat m = m2 * mask1;
       for (int col = 0; col < m.cols; col++) {
           m.at<float>(0, col) = Common_API::sigmoid_function(m.at<float>(0, col));
       }
       cv::Mat m1 = m.reshape(1, 160);
       int mx1 = std::max(0, int((x1 * sx) / x_factor));
       int mx2 = std::max(0, int((x2 * sx) / x_factor));
       int my1 = std::max(0, int((y1 * sy) / y_factor));
       int my2 = std::max(0, int((y2 * sy) / y_factor));

       // fix out of range box boundary on 2022-12-14
       if (mx2 >= m1.cols) {
           mx2 = m1.cols - 1;
       }
       if (my2 >= m1.rows) {
           my2 = m1.rows - 1;
       }
       // end fix it!!

       cv::Mat mask_roi = m1(cv::Range(my1, my2), cv::Range(mx1, mx2));
       cv::Mat rm, det_mask;
       cv::resize(mask_roi, rm, cv::Size(x2 - x1, y2 - y1));
       for (int r = 0; r < rm.rows; r++) {
           for (int c = 0; c < rm.cols; c++) {
               float pv = rm.at<float>(r, c);
               if (pv > 0.5) {
                   rm.at<float>(r, c) = 1.0;
               }
               else {
                   rm.at<float>(r, c) = 0.0;
               }
           }
       }
       rm = rm * rng.uniform(0, 255);
       rm.convertTo(det_mask, CV_8UC1);
       if ((y1 + det_mask.rows) >= inputimage.rows) {
           y2 = inputimage.rows - 1;
       }
       if ((x1 + det_mask.cols) >= inputimage.cols) {
           x2 = inputimage.cols - 1;
       }
       // std::cout << "x1: " << x1 << " x2:" << x2 << " y1: " << y1 << " y2: " << y2 << std::endl;
       cv::Mat mask = cv::Mat::zeros(cv::Size(inputimage.cols, inputimage.rows), CV_8UC1);
       det_mask(cv::Range(0, y2 - y1), cv::Range(0, x2 - x1)).copyTo(mask(cv::Range(y1, y2), cv::Range(x1, x2)));
       add(rgb_mask, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), rgb_mask, mask);

       cv::rectangle(inputimage, boxes[idx], cv::Scalar(0,0,255), 2, 8,0);
       cv::putText(inputimage, cv::format("%s_%.2f", labels[cid].c_str(), confidences[idx]) , boxes[idx].tl(),
                   cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0,255,0), 2, 8);
   }

   // compute the fps
   float t = (cv::getTickCount() - start_time) / static_cast<float>(cv::getTickFrequency());
   cv::putText(inputimage, cv::format("FPS: %.2f", 1.0/t), cv::Point(20,40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);

   cv::Mat result;
   cv::addWeighted(inputimage, 0.5, rgb_mask, 0.5, 0, result);
   result.copyTo(inputimage);
}

void Yolov5_Seg_Openvino_Deploy::process()
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
void Yolov5_Seg_Openvino_Deploy::set_Show_image(Show *imageShower)
{
    image_show = imageShower;
}

void Yolov5_Seg_Openvino_Deploy::modelRunner()
{
    this->process();
}
