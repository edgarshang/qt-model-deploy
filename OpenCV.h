#ifndef OPENCV_H
#define OPENCV_H

#include "common_api.h"
#include <QThread>


class OpenCVDecoder : public DeCode
{
    Q_OBJECT
public:
    OpenCVDecoder(QString videoPath);
    virtual void deCodeImage() override;

signals:
    void openVideoThread(QString path);

private slots:
    void onOpenVideo(QString path);

private:
    cv::VideoCapture m_cap;
    bool m_isOpend = false;
    QThread *m_workThread = nullptr;
};

#endif // OPENCV_H
