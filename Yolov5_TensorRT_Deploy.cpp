//#include "Yolov5_Onnx_Deploy.h"
#include "Yolov5_TensorRT_Deploy.h"
#include <QDebug>

using namespace nvinfer1;
using namespace nvonnxparser;
using namespace cv;



Yolov5_TensorRT_Deploy::Yolov5_TensorRT_Deploy(modelConfInfo_ info)
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
    printf("the size is ^%d\n", numIOTensors);

    for (int i = 0; i < numIOTensors; i++)
    {
        const char* tensorName = m_cudaEngine->getIOTensorName(i);
        nvinfer1::TensorIOMode ioMode = m_cudaEngine->getTensorIOMode(tensorName);
//        std::cout << "the name is " << tensorName;
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
                    out_num = tensorDims.d[j];
                }else if(j == 2)
                {
                    out_ch = tensorDims.d[j];
                }


            }
        }


        std::cout << std::endl;
    }

    std::cout << "create the context: " << std::endl;
    // 创建执行上下文
    m_context = m_cudaEngine->createExecutionContext();

    outputSize = 85*25200;
//    prob.resize(outputSize);
    cudaMallocHost(&prob, outputSize * sizeof(float));
    cudaMallocHost(&inputHost, input_h* input_w * 3 * sizeof(float));
    cudaMalloc(&buffers[0], input_h* input_w * 3 * sizeof(float));
    cudaMalloc(&buffers[1], outputSize * sizeof(float));

    m_context->setTensorAddress("images", buffers[0]);
    m_context->setTensorAddress("output0", buffers[1]);

    cudaStreamCreate(&stream);

}

Yolov5_TensorRT_Deploy::~Yolov5_TensorRT_Deploy()
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

    if(trtModeStream != nullptr)
    {
        delete[] trtModeStream;
        trtModeStream = nullptr;
    }

    cudaStreamDestroy(stream);

    if(inputHost != nullptr)
    {
        cudaFreeHost(inputHost);
    }

    if(prob != nullptr)
    {
        cudaFreeHost(prob);
    }



}

void Yolov5_TensorRT_Deploy::get_model_info()
{


}


cv::Mat Yolov5_TensorRT_Deploy::pre_image_process(cv::Mat &image)
{
    start_time = cv::getTickCount();
    int w = image.cols;
    int h = image.rows;

    int _max = std::max(h,w);

    cv::Mat image_m = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
    cv::Rect roi(0,0,w,h);
    image.copyTo(image_m(roi));
    x_factor = image_m.cols / static_cast<float>(input_h);
    y_factor = image_m.rows / static_cast<float>(input_w);

    cv::Mat blob = cv::dnn::blobFromImage(image_m, 1.0/255.0, cv::Size(input_w, input_h),
                                          cv::Scalar(0,0,0), true, true);

    return blob;
}
void Yolov5_TensorRT_Deploy::run_model(cv::Mat &input_image)
{
    start_time = cv::getTickCount();
    memcpy(inputHost, input_image.ptr<float>(), input_h*input_w*3*sizeof(float));
    cudaMemcpyAsync(buffers[0], inputHost, input_h*input_w*3*sizeof(float), cudaMemcpyHostToDevice, stream);
    m_context->enqueueV3(stream);
}

void Yolov5_TensorRT_Deploy::post_image_process(cv::Mat &inputimage)
{
    cudaMemcpyAsync(prob, buffers[1], outputSize*sizeof(float), cudaMemcpyDeviceToHost, stream);
//    cudaStreamSynchronize(stream);
    end_time = cv::getTickCount();
    float *pdata = prob;
    // 后处理 1x25200x85 85-box conf 80- min/max
    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;

    cv::Mat det_output(out_num, out_ch, CV_32F, (float*)pdata);

    det_output = (model == YOLOV5 ? det_output : det_output.t());
//    qDebug() << "det_output.rows == " << det_output.rows;

    for(int i = 0; i < det_output.rows; i++)
    {
        if (model == YOLOV5)
        {
            float conf = det_output.at<float>(i,4);
            if(conf < 0.45)
            {
                continue;
            }
        }


        cv::Mat classes_scores = det_output.row(i).colRange((model == YOLOV5 ? 5 : 4), (model == YOLOV5 ? out_ch : out_num));
        cv::Point classIdPoint;
        double score;
        cv::minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

        // 置信度0-1之间
        if( score > 0.25)
        {
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
        }
    }

    // NMS
    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, (float)(0.25), (float)(0.45), indexes);
    for(size_t i = 0; i < indexes.size(); i++)
    {
        int idx = indexes[i];
        int cid = classIds[idx];
        cv::rectangle(inputimage, boxes[idx], cv::Scalar(0,0,255), 2, 8,0);
        cv::putText(inputimage, cv::format("%s_%.2f", labels[cid].c_str(), confidences[idx]) , boxes[idx].tl(),
                    cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0,255,0), 2, 8);
    }

    // compute the fps
    float t = (end_time - start_time) / static_cast<float>(cv::getTickFrequency());
    cv::putText(inputimage, cv::format("FPS: %.2f", 1.0/t), cv::Point(20,40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
}

void Yolov5_TensorRT_Deploy::modelStop()
{
    m_runingFlag = false;
}

void Yolov5_TensorRT_Deploy::process()
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
//                emit FrameReady(frame);
//                qDebug() << "hell";
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
void Yolov5_TensorRT_Deploy::set_Show_image(Show *imageShower)
{
    image_show = imageShower;
}

void Yolov5_TensorRT_Deploy::modelRunner()
{
    this->process();
}
