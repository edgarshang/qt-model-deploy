#include "OpenCV.h"
#include <QDebug>

OpenCVDecoder::OpenCVDecoder(QString videoPath): DeCode(videoPath)
{
    qDebug() << "OpenCV instruct";
    m_workThread = new QThread();
    this->moveToThread(m_workThread);
    connect(this, &DeCode::openVideoThread, this, &OpenCVDecoder::onOpenVideo, Qt::QueuedConnection);

    m_workThread->start();
}

void OpenCVDecoder::onOpenVideo(QString path)
{
//    Q_UNUSED(path);
    qDebug() << "子线程线程ID:" << QThread::currentThreadId();
    qDebug() << "the new thread is here";
    m_cap.open(path.toStdString());
    if(m_cap.isOpened())
    {
        m_isOpend = true;
        qDebug() << "the video path is " << path;
        double fps = m_cap.get(cv::CAP_PROP_FPS);

        cv::Mat frame;
        m_cap >> frame;
        while(!frame.empty())
        {
            // call the inference function
            if(ModelInfer)
            {
                ModelInfer->inference(frame);
            }

            // call the shower
            emit frameReady(frame);

            int delay = static_cast<int>(1000/fps);
            QThread::msleep(delay);
            m_cap >> frame;
        }
    }
}

void OpenCVDecoder::deCodeImage()
{
    emit openVideoThread(videoPath);
}
