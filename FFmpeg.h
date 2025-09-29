#ifndef FFMPEG_H
#define FFMPEG_H

#include "common_api.h"
#include <QThread>

extern "C" {
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavcodec/avcodec.h>
}

class FFmpegDecoder : public DeCode
{
    Q_OBJECT
public:
    FFmpegDecoder(QString videoPath);
    virtual void deCodeImage() override;

private slots:
    void onOpenVideo(QString path);

private:
    AVFormatContext *formatContext = nullptr;
    AVCodecContext *codecContext = nullptr;
    SwsContext *swsContext = nullptr;
    AVFrame *frame = nullptr;
    AVPacket *packet = nullptr;
    int videoStreamIndex = -1;

    int m_fps = 0;
    int64_t m_frames = 0;
    int64_t m_cur_frame = 0;

};

#endif // FFMPEG_H
