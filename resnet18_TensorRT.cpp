#include "resnet18_TensorRT.h"
#include <QDebug>

using namespace nvinfer1;
using namespace nvonnxparser;
using namespace cv;

//class Logger1 :public ILogger
//{
//    void  log(Severity severity, const char* msg) noexcept
//    {
//        if (severity != Severity::kINFO)
//        {
//            std::cout << msg << std::endl;
//        }
//    }
//}gLogger;


std::string labels_txt_file = "D:/project/ort-deploy/imagenet_classes.txt";
std::vector<std::string> readClassNames();
std::vector<std::string> readClassNames()
{
    std::vector<std::string> classNames;

    std::ifstream fp(labels_txt_file);
    if (!fp.is_open())
    {
        printf("could not open file...\n");
        exit(-1);
    }
    std::string name;
    while (!fp.eof())
    {
        std::getline(fp, name);
        if (name.length())
            classNames.push_back(name);
    }
    fp.close();
    return classNames;
}

resnet18_TensorRT::resnet18_TensorRT(modelConfInfo_ info)
{
    printf("hello, wrold\n");
    label_path = info.label_text;
    model_path = info.modelPath;

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
    nvinfer1::IExecutionContext *context = m_cudaEngine->createExecutionContext();

    m_inputH = 224;
    m_inputW = 224;
    m_outputSize = 1000;

    cudaMalloc(&buffers[0], m_inputH* m_inputW * 3 * sizeof(float));
    cudaMalloc(&buffers[1], m_outputSize * sizeof(float));

    // 图像预处理
    cv::Mat image = cv::imread("D:/project/OpenCV/opencvcode/sources/doc/tutorials/dnn/images/space_shuttle.jpg");
    cv::Mat rgb, blob;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, blob, cv::Size(224, 224));
    blob.convertTo(blob, CV_32F, 1.0 / 255);
    cv::subtract(blob, cv::Scalar(0.485, 0.456, 0.406), blob);
    cv::divide(blob, cv::Scalar(0.229, 0.224, 0.225), blob);

    // HWC->CHW
    cv::Mat tensor = cv::dnn::blobFromImage(blob);

    // copy data to the GPU
    cudaMemcpy(buffers[0], tensor.ptr<float>(), 224*224*3*sizeof(float), cudaMemcpyHostToDevice);


    auto start = std::chrono::high_resolution_clock::now();
    // infer
    context->executeV2(buffers);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> inference_time = end - start;

    std::cout << "infer time :" << inference_time.count() << " ms" << std::endl;

    // copy data from GPU
    std::vector<float> output(m_outputSize);
    cudaMemcpy(output.data(), buffers[1], m_outputSize*sizeof(float), cudaMemcpyDeviceToHost);

    auto maxId = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    std::cout << "maxId = " << maxId;
    std::cout << "the object is " << labels[maxId] << std::endl;

    cv::putText(image, labels[maxId], cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2, 8);
    cv::imshow("输入图像", image);
    cv::waitKey(0);



    // 释放资源
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
}

resnet18_TensorRT::~resnet18_TensorRT()
{

}

void resnet18_TensorRT::get_model_info()
{

}

cv::Mat resnet18_TensorRT::pre_image_process(cv::Mat &image)
{
//    start_time = cv::getTickCount();
    // set input image
    cv::Mat rgb, blob;
    // RGB order
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, blob, cv::Size(224, 224));
    blob.convertTo(blob, CV_32F);
    blob = blob / 255.0;
    cv::subtract(blob, cv::Scalar(0.485, 0.456, 0.406), blob);
    cv::divide(blob, cv::Scalar(0.229, 0.224, 0.225), blob);

    return blob;
}

void resnet18_TensorRT::run_model(cv::Mat &input_image)
{
    cudaMemcpy(buffers[0], input_image.ptr<float>(), 224*224*3*sizeof(float), cudaMemcpyHostToDevice);
    m_context->executeV2(buffers);
}

void resnet18_TensorRT::post_image_process(cv::Mat &inputimage)
{
    std::vector<float> output(m_outputSize);
    cudaMemcpy(output.data(), buffers[1], m_outputSize*sizeof(float), cudaMemcpyDeviceToHost);
    auto maxId = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    cv::putText(inputimage, labels[maxId], cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2, 8);
}
void resnet18_TensorRT::process()
{
    labels = Common_API::readClassNames(label_path);
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
//    this->process();
}
