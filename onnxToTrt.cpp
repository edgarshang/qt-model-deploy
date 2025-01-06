#include "onnxToTrt.h"


#include <QDebug>

using namespace nvinfer1;
using namespace nvonnxparser;
using namespace cv;



ONNX_TO_TENSORRT::ONNX_TO_TENSORRT(modelConfInfo_ info)
{

    printf("hello, wrold\n");
    label_path = info.label_text;
    model_path = info.modelPath;
    image_path = info.imagePath;
    model = info.modelType;

    qDebug() << "the model path is " << model_path.c_str();

    labels = Common_API::readClassNames(label_path);
    m_builder = createInferBuilder(m_loger);
    m_builder->getLogger()->log(nvinfer1::ILogger::Severity::kERROR, "Create Builder...");

    m_buildConfig = m_builder->createBuilderConfig();
    m_network = m_builder->createNetworkV2(0U);
    m_parser = createParser(*m_network, m_loger);
    if(!m_parser->parseFromFile(model_path.c_str(), static_cast<int>(ILogger::Severity::kINTERNAL_ERROR)))
    {
        qDebug() << "ERROR: unable to parse onnx model.";
    }

    // 需要更改shape的时候，需要调用以下两句
    ITensor * inputTensor = m_network->getInput(0);
    inputTensor->setDimensions(Dims4{1,3,320,320});

    m_cudaEngine = m_builder->buildEngineWithConfig(*m_network, *m_buildConfig);

    IHostMemory* serializeModel = m_cudaEngine->serialize();

    std::ofstream engineFileOut("D:/project/ort-deploy/test-code.engine", std::ios::binary);

    engineFileOut.write(static_cast<const char*>(serializeModel->data()), serializeModel->size());

    engineFileOut.close();

}

ONNX_TO_TENSORRT::~ONNX_TO_TENSORRT()
{
    std::cout << "disconstruct" << std::endl;

}

void ONNX_TO_TENSORRT::get_model_info()
{


}


cv::Mat ONNX_TO_TENSORRT::pre_image_process(cv::Mat &image)
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
void ONNX_TO_TENSORRT::run_model(cv::Mat &input_image)
{
//    cudaMemcpy(buffers[0], input_image.ptr<float>(), input_h*input_w*3*sizeof(float), cudaMemcpyHostToDevice);
//    m_context->executeV2(buffers);

    cudaMemcpyAsync(buffers[0], input_image.ptr<float>(), input_h*input_w*1*sizeof(float), cudaMemcpyHostToDevice, stream);
    m_context->enqueueV3(stream);
}


void ONNX_TO_TENSORRT::post_image_process(cv::Mat &inputimage)
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

void ONNX_TO_TENSORRT::modelStop()
{
    m_runingFlag = false;
}

void ONNX_TO_TENSORRT::process()
{


}
// show
void ONNX_TO_TENSORRT::set_Show_image(Show *imageShower)
{
    image_show = imageShower;
}

void ONNX_TO_TENSORRT::modelRunner()
{
    this->process();
}

