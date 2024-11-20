#include "resnet18_TensorRT.h"
#include <QDebug>

resnet18_TensorRT::resnet18_TensorRT(modelConfInfo_ info)
{
    m_build = createInferBuilder(m_loger);
    model_path = "D:/project/ort-deploy/resnet18.engine";
    label_path = info.label_text;

    qDebug() << "path = " << QString::fromStdString(model_path);
    std::ifstream file(model_path, std::ios::binary);
    char* trtModeStream = nullptr;
    int size = 0;
    if(file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        trtModeStream = new char[size];
        assert(trtModeStream);
        file.read(trtModeStream, size);
        file.close();
    }

    m_runtime = createInferRuntime(m_loger);
    m_cudaEngine = m_runtime->deserializeCudaEngine(trtModeStream, size);
    int numIOTensors = m_cudaEngine->getNbIOTensors();
    qDebug() <<"the size is " << numIOTensors;

    for(int i = 0; i < numIOTensors; i++)
    {
        const char* tensorName = m_cudaEngine->getIOTensorName(i);
        nvinfer1::TensorIOMode ioMode = m_cudaEngine->getTensorIOMode(tensorName);
        qDebug() << "the name is " << tensorName;
        qDebug() << "Tensor I/O mode : " << (ioMode == nvinfer1::TensorIOMode::kINPUT ? "Input" : "Output");


        nvinfer1::Dims tensorDims = m_cudaEngine->getTensorShape(tensorName);
        qDebug() << "Tensor Dimensions: ";
        for(int j = 0; j < tensorDims.nbDims; j++)
        {
            qDebug() << tensorDims.d[j];
        }
    }

    qDebug() << "create the context: ";
    // 创建执行上下文
    m_context = m_cudaEngine->createExecutionContext();

    cudaMalloc(&buffers[0], m_inputW*m_inputW*3*sizeof (float));
    cudaMalloc(&buffers[1], m_outputSize * sizeof (float));
}

resnet18_TensorRT::~resnet18_TensorRT()
{

}

void resnet18_TensorRT::get_model_info()
{

}

cv::Mat resnet18_TensorRT::pre_image_process(cv::Mat &image)
{
    start_time = cv::getTickCount();
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
    this->process();
}
