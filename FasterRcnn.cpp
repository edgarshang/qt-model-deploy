#include "FasterRcnn.h"

FasterRcnn::FasterRcnn(std::string modelPath, std::string imagePath, std::string label_text, std::string modelType)
{
    model_path = modelPath;
    image_path = imagePath;
    label_path = label_text;
    model = modelType;

    std::cout << "hello, world";
}

FasterRcnn::~FasterRcnn()
{



}

void FasterRcnn::get_model_info()
{
    env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "faster-rcnn-onnx");
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

cv::Mat FasterRcnn::pre_image_process(cv::Mat &image)
{
    start_time = cv::getTickCount();
    input_h = image.rows;
    input_w = image.cols;
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0 / 255.0, cv::Size(image.cols, image.rows), cv::Scalar(0, 0, 0), true, false);

    return blob;
}

void FasterRcnn::run_model(cv::Mat &input_image)
{
    std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };
    size_t tpixels = input_h * input_w * 3;

    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, input_image.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
    const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
    const std::array<const char*, 3> outNames = { output_node_names[0].c_str(), output_node_names[1].c_str(), output_node_names[2].c_str() };

    try {
        ort_outputs = session_->Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
    }
    catch (std::exception e) {
        std::cout << e.what() << std::endl;
    }
}

void FasterRcnn::post_image_process(std::vector<Ort::Value> &outputs, cv::Mat &inputimage)
{
    const float* boxes = outputs[0].GetTensorMutableData<float>();
    const int64* labels = nullptr;
    const float* scores = nullptr;
    if(model == "FasterRcnn")
    {
        labels = outputs[1].GetTensorMutableData<int64>();
        scores = outputs[2].GetTensorMutableData<float>();
    }else if(model == "RetinaNet")
    {
        labels = outputs[2].GetTensorMutableData<int64>();
        scores = outputs[1].GetTensorMutableData<float>();
    }



    auto outShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t rows = outShape[0];

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
        }
    }

    // compute the fps
    float t = (cv::getTickCount() - start_time) / static_cast<float>(cv::getTickFrequency());
    cv::putText(inputimage, cv::format("FPS: %.2f", 1.0/t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255,0,0), 2, 8);
}

void FasterRcnn::process()
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
void FasterRcnn::set_Show_image(Show *imageShower)
{
    image_show = imageShower;
}

void FasterRcnn::modelRunner()
{
    this->process();
}
