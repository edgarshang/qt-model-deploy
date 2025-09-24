#include "Yolov5_Seg_TensorRT_Deploy.h"
#include <QDebug>

using namespace nvinfer1;
using namespace nvonnxparser;
using namespace cv;



Yolov5_Seg_TensorRT_Deploy::Yolov5_Seg_TensorRT_Deploy(modelConfInfo_ info)
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

    sx = 160.0f / 640.0f;
       sy = 160.0f / 640.0f;

    for (int i = 0; i < numIOTensors; i++)
    {
        const char* tensorName = m_cudaEngine->getIOTensorName(i);
//        nvinfer1::TensorIOMode ioMode = m_cudaEngine->getTensorIOMode(tensorName);
        std::cout << "the name is " << tensorName << ": ";
//        std::cout << " Tensor I/O Mode : " << (ioMode == nvinfer1::TensorIOMode::kINPUT ? "Input" : "Output") << std::endl;

        nvinfer1::Dims tensorDims = m_cudaEngine->getTensorShape(tensorName); // 获取张量维度
//        std::cout << "Tensor Dimensions: ";
        if(QString(tensorName) == "images")
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
        }else if(QString(tensorName) == "output0")
        {
            for (int j = 0; j < tensorDims.nbDims; j++)
            {
                std::cout << tensorDims.d[j] << " ";   // 打印出张量维度

                 outputSize0 *= tensorDims.d[j];
                 if(j == 1)
                 {
                     out_num = tensorDims.d[j];
                 }else if(j == 2)
                 {
                     out_ch = tensorDims.d[j];
                 }

            }
        }else if(QString(tensorName) == "output1")
        {
            for (int j = 0; j < tensorDims.nbDims; j++)
            {
                std::cout << tensorDims.d[j] << " ";   // 打印出张量维度

                outputSize1 *= tensorDims.d[j];
            }
        }


        std::cout << std::endl;
    }

    std::cout << "create the context: " << std::endl;
    // 创建执行上下文
    m_context = m_cudaEngine->createExecutionContext();

    qDebug() << "input_h = " << input_h << " " << "input_w = " << input_w;
    qDebug() << "outputSize0 = " << outputSize0 << " " << "outputSize1 = " << outputSize1;
    cudaMalloc(&buffers[0], input_h* input_w * 3 * sizeof(float));
    cudaMalloc(&buffers[1], outputSize0 * sizeof(float));
    cudaMalloc(&buffers[2], outputSize1 * sizeof(float));

}

Yolov5_Seg_TensorRT_Deploy::~Yolov5_Seg_TensorRT_Deploy()
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
}

void Yolov5_Seg_TensorRT_Deploy::get_model_info()
{


}


cv::Mat Yolov5_Seg_TensorRT_Deploy::pre_image_process(cv::Mat &image)
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
void Yolov5_Seg_TensorRT_Deploy::run_model(cv::Mat &input_image)
{
    cudaMemcpy(buffers[0], input_image.ptr<float>(), input_h*input_w*3*sizeof(float), cudaMemcpyHostToDevice);
    m_context->executeV2(buffers);
}

void Yolov5_Seg_TensorRT_Deploy::post_image_process(cv::Mat &inputimage)
{
//    qDebug() << "the outputSize = " << outputSize;
    std::vector<float> output(outputSize0);
    std::vector<float> output1(outputSize1);
    cudaMemcpy(output.data(), buffers[1], outputSize0*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output1.data(), buffers[2], outputSize1*sizeof(float), cudaMemcpyDeviceToHost);

    float *pdata = output.data();
    float *mdata = output1.data();
    // 后处理 1x25200x85 85-box conf 80- min/max
    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;

    std::vector<cv::Mat> masks;
    cv::Mat mask1(32, 25600, CV_32F, (float*)mdata);

    cv::Mat det_output(out_num, out_ch, CV_32F, (float*)pdata);

    det_output = (model == YOLOV5_SEG ? det_output : det_output.t());
//    qDebug() << "det_output.rows == " << det_output.rows;

    for(int i = 0; i < det_output.rows; i++)
    {
        if (model == YOLOV5_SEG)
        {
            float conf = det_output.at<float>(i,4);
            if(conf < 0.45)
            {
                continue;
            }
        }


        cv::Mat classes_scores = det_output.row(i).colRange((model == YOLOV5_SEG ? 5 : 4), (model == YOLOV5_SEG ? out_ch - 32: out_num - 32));
        cv::Point classIdPoint;
        double score;
        cv::minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

        // 置信度0-1之间
        if( score > 0.25)
        {
            cv::Mat mask2 = det_output.row(i).colRange(model == YOLOV5_SEG ? (out_ch - 32): (out_num - 32), model == YOLOV5_SEG ? (out_ch) : (out_num));
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
            masks.push_back(mask2);
        }
    }

    // NMS
    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, (float)(0.25), (float)(0.45), indexes);
    cv::Mat rgb_mask = cv::Mat::zeros(inputimage.size(), inputimage.type());
    for(size_t i = 0; i < indexes.size(); i++)
    {
        int idx = indexes[i];
        int cid = classIds[idx];

        cv::Rect box = boxes[idx];
        int x1 = std::max(0, box.x);
        int y1 = std::max(0, box.y);
        int x2 = std::max(0, box.br().x);
        int y2 = std::max(0, box.br().y);
        cv::Mat m2 = masks[idx];
        cv::Mat m = m2 * mask1;
        for (int col = 0; col < m.cols; col++) {
            m.at<float>(0, col) = Common_API::sigmoid_function(m.at<float>(0, col));
        }
        cv::Mat m1 = m.reshape(1, 160);
        int mx1 = std::max(0, int((x1 * sx) / x_factor));
        int mx2 = std::max(0, int((x2 * sx) / x_factor));
        int my1 = std::max(0, int((y1 * sy) / y_factor));
        int my2 = std::max(0, int((y2 * sy) / y_factor));
        std::cout << "sx is " << sx << " sy is " << sy << std::endl;
                std::cout << "mx1 is " << mx1 << " mx2 is " << mx2 << " my1 is " << my1 << " my2 is " << my2 << std::endl;
                std::cout << "x1 is " << x1 << " x2 is " << x2 << " y1 is " << y1 << " y2 is " << y2 << std::endl;
                std::cout << "x_factor is " << x_factor << " y_factor is " << y_factor << std::endl;

        // fix out of range box boundary on 2022-12-14
        if (mx2 >= m1.cols) {
            mx2 = m1.cols - 1;
        }
        if (my2 >= m1.rows) {
            my2 = m1.rows - 1;
        }
        // end fix it!!

        cv::Mat mask_roi = m1(cv::Range(my1, my2), cv::Range(mx1, mx2));
        cv::Mat rm, det_mask;

        cv::resize(mask_roi, rm, cv::Size(x2 - x1, y2 - y1));
        for (int r = 0; r < rm.rows; r++) {
            for (int c = 0; c < rm.cols; c++) {
                float pv = rm.at<float>(r, c);
                if (pv > 0.5) {
                    rm.at<float>(r, c) = 1.0;
                }
                else {
                    rm.at<float>(r, c) = 0.0;
                }
            }
        }
        rm = rm * rng.uniform(0, 255);
        rm.convertTo(det_mask, CV_8UC1);
        if ((y1 + det_mask.rows) >= inputimage.rows) {
            y2 = inputimage.rows - 1;
        }
        if ((x1 + det_mask.cols) >= inputimage.cols) {
            x2 = inputimage.cols - 1;
        }
        // std::cout << "x1: " << x1 << " x2:" << x2 << " y1: " << y1 << " y2: " << y2 << std::endl;
        cv::Mat mask = cv::Mat::zeros(cv::Size(inputimage.cols, inputimage.rows), CV_8UC1);
        det_mask(cv::Range(0, y2 - y1), cv::Range(0, x2 - x1)).copyTo(mask(cv::Range(y1, y2), cv::Range(x1, x2)));
        add(rgb_mask, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), rgb_mask, mask);

        cv::rectangle(inputimage, boxes[idx], cv::Scalar(0,0,255), 2, 8,0);
        cv::putText(inputimage, cv::format("%s_%.2f", labels[cid].c_str(), confidences[idx]) , boxes[idx].tl(),
                    cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0,255,0), 2, 8);
    }

    // compute the fps
    float t = (cv::getTickCount() - start_time) / static_cast<float>(cv::getTickFrequency());
    cv::putText(inputimage, cv::format("FPS: %.2f", 1.0/t), cv::Point(20,40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);

    cv::Mat result;
   cv::addWeighted(inputimage, 0.5, rgb_mask, 0.5, 0, result);
   result.copyTo(inputimage);
}

void Yolov5_Seg_TensorRT_Deploy::modelStop()
{
    m_runingFlag = false;
}

void Yolov5_Seg_TensorRT_Deploy::process()
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
        this->post_image_process(image);
        image_show->imageshow(image);
    }

}
// show
void Yolov5_Seg_TensorRT_Deploy::set_Show_image(Show *imageShower)
{
    image_show = imageShower;
}

void Yolov5_Seg_TensorRT_Deploy::modelRunner()
{
    this->process();
}
