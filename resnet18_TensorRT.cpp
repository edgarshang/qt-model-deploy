#include "resnet18_TensorRT.h"
#include <QDebug>

using namespace nvinfer1;
using namespace nvonnxparser;
using namespace cv;

resnet18_TensorRT::resnet18_TensorRT(modelConfInfo_ info)
{
    printf("hello, wrold\n");
    label_path = info.label_text;
    model_path = info.modelPath;
    image_path = info.imagePath;

    labels = Common_API::readClassNames(label_path);
    builder = createInferBuilder(m_loger);
    builder->getLogger()->log(nvinfer1::ILogger::Severity::kERROR, "Create Builder...");

    std::ifstream file(model_path, std::ios::binary);
    int size = 0;
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModeStream = new char[size];
        assert(trtModeStream);
        file.read(trtModeStream, size);
        file.close();
    }

    m_runtime = createInferRuntime(m_loger);
    m_cudaEngine = m_runtime->deserializeCudaEngine(trtModeStream, size);
    int numIOTensors = m_cudaEngine->getNbIOTensors();
    printf("the size is ^%d\n", numIOTensors);

    for (int i = 0; i < numIOTensors; i++)
    {
        const char* tensorName = m_cudaEngine->getIOTensorName(i);
        nvinfer1::TensorIOMode ioMode = m_cudaEngine->getTensorIOMode(tensorName);
        std::cout << "the name is " << tensorName;
        std::cout << " Tensor I/O Mode : " << (ioMode == nvinfer1::TensorIOMode::kINPUT ? "Input" : "Output") << std::endl;

        nvinfer1::Dims tensorDims = m_cudaEngine->getTensorShape(tensorName); // 获取张量维度
        std::cout << "Tensor Dimensions: ";
        for (int j = 0; j < tensorDims.nbDims; j++)
        {
            std::cout << tensorDims.d[j] << " ";   // 打印出张量维度
        }

        std::cout << std::endl;
    }

    std::cout << "create the context: " << std::endl;
    // 创建执行上下文
    m_context = m_cudaEngine->createExecutionContext();

    m_inputH = 224;
    m_inputW = 224;
    m_outputSize = 1000;

    cudaMalloc(&buffers[0], m_inputH* m_inputW * 3 * sizeof(float));
    cudaMalloc(&buffers[1], m_outputSize * sizeof(float));
}

resnet18_TensorRT::~resnet18_TensorRT()
{
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
}

void resnet18_TensorRT::get_model_info()
{

}

cv::Mat resnet18_TensorRT::pre_image_process(cv::Mat &image)
{
    start_time = cv::getTickCount();
    cv::Mat rgb, blob;
    // RGB order
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, blob, cv::Size(224, 224));
    blob.convertTo(blob, CV_32F, 1.0 / 255);
    cv::subtract(blob, cv::Scalar(0.485, 0.456, 0.406), blob);
    cv::divide(blob, cv::Scalar(0.229, 0.224, 0.225), blob);

    return blob;
}

void resnet18_TensorRT::run_model(cv::Mat &input_image)
{
    cv::Mat tensor = cv::dnn::blobFromImage(input_image);
    cudaMemcpy(buffers[0], tensor.ptr<float>(), 224*224*3*sizeof(float), cudaMemcpyHostToDevice);
    m_context->executeV2(buffers);
}

void resnet18_TensorRT::post_image_process(cv::Mat &inputimage)
{
    std::vector<float> output(m_outputSize);
    cudaMemcpy(output.data(), buffers[1], m_outputSize*sizeof(float), cudaMemcpyDeviceToHost);
    auto maxId = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    cv::putText(inputimage, labels[maxId], cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2, 8);

    float t = (cv::getTickCount() - start_time) / static_cast<float>(cv::getTickFrequency());
    cv::putText(inputimage, cv::format("FPS: %.2f", 1.0/t), cv::Point(20,30), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);

}
void resnet18_TensorRT::process()
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
    }else
    {
        cv::Mat image = cv::imread(path.toStdString());
        cv::Mat model_input = this->pre_image_process(image);
        this->run_model(model_input);
        this->post_image_process(image);
        image_show->imageshow(image);
    }
}
// show
void resnet18_TensorRT::set_Show_image(Show *imageShower)
{
    image_show = imageShower;
}

void resnet18_TensorRT::modelRunner()
{
    this->process();
}
