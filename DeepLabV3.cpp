#include "DeepLabV3.h"
#include "QDebug"


DeepLabV3::DeepLabV3(std::string modelPath, std::string imagePath, std::string label_text, std::string modelType)
{
    model_path = modelPath;
    image_path = imagePath;
    label_path = label_text;
    model = modelType;

    std::cout << "hello, world";
}

DeepLabV3::~DeepLabV3()
{



}

void DeepLabV3::get_model_info()
{
    env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "deeplabv3-onnx");
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    w_model_path = std::wstring(model_path.begin(), model_path.end());

    std::cout << "onnxruntime inference try to use CPU Device" << std::endl;
    //OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
    OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 0);

    session_ = new Ort::Session(env, w_model_path.c_str(), session_options);


    int input_nodes_num = static_cast<int>(session_->GetInputCount());
    int output_nodes_num = static_cast<int>(session_->GetOutputCount());

    std::cout << "input_nodes_num : " << input_nodes_num << std::endl;
    std::cout << "output_nodes_num : " << output_nodes_num << std::endl;

    for (int i = 0; i < input_nodes_num; i++) {
        auto input_name = session_->GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());
    }

    for (int i = 0; i < output_nodes_num; i++) {
        auto output_name = session_->GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(output_name.get());
    }
}

cv::Mat DeepLabV3::pre_image_process(cv::Mat &image)
{
    start_time = cv::getTickCount();
    input_h = image.rows;
    input_w = image.cols;
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(image.cols, image.rows), cv::Scalar(0, 0, 0), true, false);

    return blob;
}

void DeepLabV3::run_model(cv::Mat &input_image)
{
    std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };
    size_t tpixels = input_h * input_w * 3;

    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, input_image.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
    const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
    const std::array<const char*, 2> outNames = { output_node_names[0].c_str(), output_node_names[1].c_str()};

    try {
        ort_outputs = session_->Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
    }
    catch (std::exception e) {
        std::cout << e.what() << std::endl;
    }
}

void DeepLabV3::post_image_process(std::vector<Ort::Value> &outputs, cv::Mat &inputimage)
{
    std::vector<cv::Vec3b> color_table;
    color_table.push_back(cv::Vec3b(0, 0, 0));
    cv::RNG rng(cv::getTickCount());
    const float* mask_data = outputs[0].GetTensorMutableData<float>();

    auto outShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
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

void DeepLabV3::process()
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
void DeepLabV3::set_Show_image(Show *imageShower)
{
    image_show = imageShower;
}

void DeepLabV3::modelRunner()
{
    this->process();
}
