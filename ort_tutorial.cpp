#include "ort_tutorial.h"

ort_tutorial::ort_tutorial(std::string modelPath, std::string imagePath, std::string label_text)
{
    model_path = modelPath;
    image_path = imagePath;
    label_path = label_text;
}


cv::Mat ort_tutorial::pre_image_process(cv::Mat &image)
{
    //cv::Mat image = cv::imread(image_path);
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

    cv::Mat input_image = cv::dnn::blobFromImage(blob);

    return input_image;
}

void ort_tutorial::run_model(cv::Mat &input_image)
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

    //return ort_outputs;
}

void ort_tutorial::post_image_process(std::vector<Ort::Value> &outputs, cv::Mat &inputimage)
{
    // 1x 1000
    const float* pdata = outputs[0].GetTensorMutableData<float>();
    cv::Mat prob(out_num, out_ch, CV_32F, (float*)pdata);

    cv::Point maxL, minL;
    double maxv, minv;
    cv::minMaxLoc(prob, &minv, &maxv, &minL, &maxL);
    int max_index = maxL.x;
//    std::cout << "label id: " << max_index << std::endl;
    cv::putText(inputimage, labels[max_index], cv::Point(50, 80), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2, 8);

    float t = (cv::getTickCount() - start_time) / static_cast<float>(cv::getTickFrequency());
    cv::putText(inputimage, cv::format("FPS: %.2f", 1.0/t), cv::Point(20,30), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);

    image_show->imageshow(inputimage);

}


void ort_tutorial::modelRunner()
{
    this->process();
}

void ort_tutorial::process()
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
    }else
    {
        cv::Mat image = cv::imread(path.toStdString());
        cv::Mat model_input = this->pre_image_process(image);
        this->run_model(model_input);
        this->post_image_process(ort_outputs, image);
        image_show->imageshow(image);
    }

    session_options.release();
    session_->release();
}

ort_tutorial::~ort_tutorial()
{
    std::cout << "disconstruct" << std::endl;
}


void ort_tutorial::get_model_info()
{
    env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "resnet18-onnx");
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
        int ch = inputShapeInfo[1];
        input_h = inputShapeInfo[2];
        input_w = inputShapeInfo[3];
        std::cout << "input format: " << ch << "x" << input_h << "x" << input_w << std::endl;
    }

    for (int i = 0; i < output_nodes_num; i++) {
        auto output_name = session_->GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(output_name.get());
        auto outShapeInfo = session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        out_num = outShapeInfo[0];
        out_ch = outShapeInfo[1];
        std::cout << "output format: " << out_num << "x" << out_ch << std::endl;
    }
}

void ort_tutorial::set_Show_image(Show *imageShower)
{
    image_show = imageShower;
}

