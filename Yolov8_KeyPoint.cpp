#include "Yolov8_KeyPoint.h"
#include <QDebug>


Yolov8_KeyPoint::Yolov8_KeyPoint(std::string modelPath, std::string imagePath, std::string label_text, std::string modelType)
{
    model_path = modelPath;
    image_path = imagePath;
    label_path = label_text;
    model = modelType;
}

Yolov8_KeyPoint::~Yolov8_KeyPoint()
{
    std::cout << "disconstruct" << std::endl;
}

void Yolov8_KeyPoint::get_model_info()
{
    env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Yolov8-pose");
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    w_model_path = std::wstring(model_path.begin(), model_path.end());

    std::cout << "onnxruntime inference try to use CPU Device" << std::endl;
    //OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
    OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 0);

    session_ = new Ort::Session(env, w_model_path.c_str(), session_options);


    int input_nodes_num = static_cast<int>(session_->GetInputCount());
    int output_nodes_num = static_cast<int>(session_->GetOutputCount());

    for (int i = 0; i < input_nodes_num; i++) {
        auto input_name = session_->GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());
        auto inputShapeInfo = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
//        GetShape();
        int ch = inputShapeInfo[1];
        input_h = inputShapeInfo[2];
        input_w = inputShapeInfo[3];
        std::cout << "input format: " << ch << "x" << input_h << "x" << input_w << std::endl;
    }

    for (int i = 0; i < output_nodes_num; i++) {
        auto output_name = session_->GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(output_name.get());
        auto outShapeInfo = session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        out_num = outShapeInfo[1];
        out_ch = outShapeInfo[2];
        std::cout << "output format: " << out_num << "x" << out_ch << std::endl;
    }
}
cv::Mat Yolov8_KeyPoint::pre_image_process(cv::Mat &image)
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

    m1_factor = cv::Mat::zeros(cv::Size(3, 17), CV_32FC1);
    for(int i = 0; i< 17; i++)
    {
        m1_factor.at<float>(i,0) = x_factor;
        m1_factor.at<float>(i,1) = y_factor;
        m1_factor.at<float>(i,2) = 1.0f;
    }

    cv::Mat blob = cv::dnn::blobFromImage(image_m, 1.0/255.0, cv::Size(input_w, input_h),
                                          cv::Scalar(0,0,0), true, true);

    return blob;
}
void Yolov8_KeyPoint::run_model(cv::Mat &input_image)
{
    std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };
    size_t tpixels = input_h * input_w * 3;
    // set input data and inference
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, input_image.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
    const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
    const std::array<const char*, 1> outNames = { output_node_names[0].c_str() };

    //std::vector<Ort::Value> ort_outputs;
    try {
        ort_outputs = session_->Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
    }
    catch (std::exception e) {
        std::cout << e.what() << std::endl;
    }
}
void Yolov8_KeyPoint::post_image_process(std::vector<Ort::Value> &outputs, cv::Mat &inputimage)
{
    const float* pdata = outputs[0].GetTensorMutableData<float>();

    // 后处理 1x25200x85 85-box conf 80- min/max
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<cv::Mat> keypoints;
    cv::Mat det_output(out_num, out_ch, CV_32F, (float*)pdata);

    det_output = det_output.t();
    // 56 = box(4) + socres(1) + pose_keyPoint(51 = 3 x 17)
    for(int i = 0; i < det_output.rows; i++)
    {

        float conf = det_output.at<float>(i,4);
//        qDebug() << "conf = " << conf;
        if(conf < 0.45)
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
        cv::Mat pts = det_output.row(i).colRange(5, out_num);
        keypoints.push_back(pts);
    }

    // NMS
    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, (float)(0.25), (float)(0.45), indexes);
    for(size_t i = 0; i < indexes.size(); i++)
    {
        int idx = indexes[i];
        cv::rectangle(inputimage, boxes[idx], cv::Scalar(0,0,255), 2, 8,0);
        cv::putText(inputimage, cv::format("%.2f", confidences[idx]) , boxes[idx].tl(),
                    cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0,255,0), 2, 8);
        cv::Mat keyPoint = keypoints[i];
        keyPoint = keyPoint.reshape(0,17);
        cv::Mat kp;

        cv::multiply(keyPoint, m1_factor, kp);
        kp = kp.reshape(0,51);

        const float* kpts_data = &kp.at<float>(0,0);
        Common_API::draw_pose_keyPoint(kpts_data, inputimage);
    }

    // compute the fps
    float t = (cv::getTickCount() - start_time) / static_cast<float>(cv::getTickFrequency());
    cv::putText(inputimage, cv::format("FPS: %.2f", 1.0/t), cv::Point(20,40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
}

void Yolov8_KeyPoint::process()
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
                this->post_image_process(ort_outputs, frame);
                image_show->imageshow(frame);
            }

            capture.release();
        }
    }
    else{
        cv::Mat image = cv::imread(path.toStdString());
        cv::Mat model_input = this->pre_image_process(image);
        this->run_model(model_input);
        this->post_image_process(ort_outputs, image);
        image_show->imageshow(image);
    }

    session_options.release();
    session_->release();
}
// show
void Yolov8_KeyPoint::set_Show_image(Show *imageShower)
{
    image_show = imageShower;
}

void Yolov8_KeyPoint::modelRunner()
{
    this->process();
}
