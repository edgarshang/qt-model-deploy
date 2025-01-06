#include "DeepLabV3_TensorRT.h"


#include <QDebug>

using namespace nvinfer1;
using namespace nvonnxparser;
using namespace cv;



DEEPLABV3_TensorRT_Deploy::DEEPLABV3_TensorRT_Deploy(modelConfInfo_ info)
{

    printf("hello, wrold\n");
    label_path = info.label_text;
    model_path = info.modelPath;
    image_path = info.imagePath;
    model = info.modelType;

    labels = Common_API::readClassNames(label_path);
    m_builder = createInferBuilder(m_loger);
    m_builder->getLogger()->log(nvinfer1::ILogger::Severity::kERROR, "Create Builder...");

    int size = Common_API::load_tensorRT_model(&trtModeStream, model_path.c_str());
    m_runtime = createInferRuntime(m_loger);
    m_cudaEngine = m_runtime->deserializeCudaEngine(trtModeStream, size);
    int numIOTensors = m_cudaEngine->getNbIOTensors();
    printf("the size is %d\n", numIOTensors);

    m_context = m_cudaEngine->createExecutionContext();


    for (int i = 0; i < numIOTensors; i++)
    {
        const char* tensorName = m_cudaEngine->getIOTensorName(i);
        nvinfer1::TensorIOMode ioMode = m_cudaEngine->getTensorIOMode(tensorName);
        std::cout << "the name is " << tensorName;
//        std::cout << " Tensor I/O Mode : " << (ioMode == nvinfer1::TensorIOMode::kINPUT ? "Input" : "Output") << std::endl;

        nvinfer1::Dims tensorDims = m_cudaEngine->getTensorShape(tensorName); // 获取张量维度
//        std::cout << "Tensor Dimensions: ";
        if(ioMode == nvinfer1::TensorIOMode::kINPUT)
        {
            for (int j = 0; j < tensorDims.nbDims; j++)
            {
                std::cout << tensorDims.d[j] << " ";   // 打印出张量维度
                if(j == 2)
                {
                    input_h = tensorDims.d[j];
                }else if(j == 3)
                {
                    input_w = tensorDims.d[j];
                }
            }
        }else if(ioMode == nvinfer1::TensorIOMode::kOUTPUT)
        {
            for (int j = 0; j < tensorDims.nbDims; j++)
            {
                std::cout << tensorDims.d[j] << " ";   // 打印出张量维度

                if(j == 1)
                {
                    out_cn = tensorDims.d[j];
                }else if(j == 2)
                {
                    out_num = tensorDims.d[j];
                }else if(j == 3)
                {
                    out_ch = tensorDims.d[j];
                }


            }
        }


        std::cout << std::endl;
    }

    outputSize = out_num * out_ch * out_cn;
    prob.resize(outputSize);
    cudaMalloc(&buffers[0], input_h* input_w * 3 * sizeof(float));
    cudaMalloc(&buffers[1], outputSize * sizeof(float));
    cudaMalloc(&buffers[2], outputSize * sizeof(float));

    m_context->setTensorAddress("input", buffers[0]);
    m_context->setTensorAddress("out", buffers[1]);
    m_context->setTensorAddress("aux", buffers[2]);

    cudaStreamCreate(&stream);

}

DEEPLABV3_TensorRT_Deploy::~DEEPLABV3_TensorRT_Deploy()
{
    std::cout << "disconstruct" << std::endl;
    // 释放资源

    if(buffers[0] != nullptr)
    {
        cudaFree(buffers[0]);
        buffers[0] = nullptr;
    }

    if(buffers[1] != nullptr)
    {
        cudaFree(buffers[1]);
        buffers[1] = nullptr;
    }

    if(buffers[2] != nullptr)
    {
        cudaFree(buffers[2]);
        buffers[2] = nullptr;
    }

    if(trtModeStream != nullptr)
    {
        delete[] trtModeStream;
        trtModeStream = nullptr;
    }

    cudaStreamDestroy(stream);

}

void DEEPLABV3_TensorRT_Deploy::get_model_info()
{


}


cv::Mat DEEPLABV3_TensorRT_Deploy::pre_image_process(cv::Mat &image)
{

    start_time = cv::getTickCount();
//    int w = image.cols;
//    int h = image.rows;

//    int _max = std::max(h,w);

//    cv::Mat image_m = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
//    cv::Rect roi(0,0,w,h);
//    image.copyTo(image_m(roi));
//    x_factor = image_m.cols / static_cast<float>(input_h);
//    y_factor = image_m.rows / static_cast<float>(input_w);

    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0/255.0, cv::Size(input_w, input_h),
                                          cv::Scalar(0,0,0), true, true);

    return blob;
}
void DEEPLABV3_TensorRT_Deploy::run_model(cv::Mat &input_image)
{
//    cudaMemcpy(buffers[0], input_image.ptr<float>(), input_h*input_w*3*sizeof(float), cudaMemcpyHostToDevice);
//    m_context->executeV2(buffers);

    cudaMemcpyAsync(buffers[0], input_image.ptr<float>(), input_h*input_w*1*sizeof(float), cudaMemcpyHostToDevice, stream);
    m_context->enqueueV3(stream);
}


void DEEPLABV3_TensorRT_Deploy::post_image_process(cv::Mat &inputimage)
{
    cudaMemcpyAsync(prob.data(), buffers[1], outputSize*sizeof(float), cudaMemcpyDeviceToHost, stream);
    float *mask_data = prob.data();

    std::vector<cv::Vec3b> color_table;
    color_table.push_back(cv::Vec3b(0, 0, 0));
    cv::RNG rng(cv::getTickCount());


    int num_cn = out_cn;
    int out_h = out_num;
    int out_w = out_ch;
//    qDebug() <<num_cn<< "x" << out_h << "x" << out_w;

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

    cv::resize(result, result, cv::Size(inputimage.cols, inputimage.rows));
    cv::addWeighted(inputimage, 0.7, result, 0.3, 0, inputimage);
//    cv::Mat gray;
//    cv::cvtColor(result, gray, cv::COLOR_BGR2GRAY);
//    cv::Mat dst;
//    cv::bitwise_and(inputimage, inputimage, dst, gray);

    // compute the fps
    float t = (cv::getTickCount() - start_time) / static_cast<float>(cv::getTickFrequency());
    cv::putText(inputimage, cv::format("FPS: %.2f", 1.0/t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255,0,0), 2, 8);

}

void DEEPLABV3_TensorRT_Deploy::modelStop()
{
    m_runingFlag = false;
}

void DEEPLABV3_TensorRT_Deploy::process()
{
    labels = Common_API::readClassNames(label_path);

    QString path = QString::fromStdString(image_path);


    if(path.endsWith(".mp4") || path.endsWith(".avi"))
    {
        cv::VideoCapture capture(path.toStdString());
        double fps = capture.get(cv::CAP_PROP_FPS);
        int delay = static_cast<int>(1000 / fps);
        if(capture.isOpened())
        {
            cv::Mat frame;
            while(true)
            {
                if(!m_runingFlag)
                {
                    return;
                }

                bool ret = capture.read(frame);
                if(!ret)
                {
                    break;
                }
                cv::Mat model_input = this->pre_image_process(frame);
                this->run_model(model_input);
                this->post_image_process(frame);
                image_show->imageshow(frame);
                if (cv::waitKey(delay) == 27) { // 按下 ESC 键退出
                    break;
                }
            }

            capture.release();
        }
    }
    else{
        cv::Mat image = cv::imread(path.toStdString());
        cv::Mat model_input = this->pre_image_process(image);
        this->run_model(model_input);
        this->post_image_process(image);  // TODO-A
        image_show->imageshow(image);
    }

}
// show
void DEEPLABV3_TensorRT_Deploy::set_Show_image(Show *imageShower)
{
    image_show = imageShower;
}

void DEEPLABV3_TensorRT_Deploy::modelRunner()
{
    this->process();
}

