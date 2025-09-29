#include "FFmpeg.h"
#include <QDebug>
#include <QImage>

FFmpegDecoder::FFmpegDecoder(QString videoPath):DeCode(videoPath)
{
    qDebug() << "FFmepg struct";
    m_workThread = new QThread();
    this->moveToThread(m_workThread);
    connect(this, &DeCode::openVideoThread, this, &FFmpegDecoder::onOpenVideo, Qt::QueuedConnection);

    m_workThread->start();
}

void FFmpegDecoder::deCodeImage()
{
    emit openVideoThread(videoPath);
}


void FFmpegDecoder::onOpenVideo(QString path)
{
    Q_UNUSED(path);
    qDebug() << "ffmepg decode video";

    frame = av_frame_alloc();
    packet = av_packet_alloc();

    if (formatContext)
    {
        avformat_close_input(&formatContext);
        formatContext = nullptr;
    }

    avformat_open_input(&formatContext, path.toStdString().c_str(), nullptr, nullptr);
    avformat_find_stream_info(formatContext, nullptr);

    videoStreamIndex = -1;
    for (unsigned int i = 0; i < formatContext->nb_streams; i++) {
      if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
          videoStreamIndex = i;
          break;
      }
    }

    m_fps = 1000 / static_cast<uint8_t>(av_q2d(formatContext->streams[videoStreamIndex]->r_frame_rate));
    m_frames = formatContext->streams[videoStreamIndex]->nb_frames;
    qDebug() << "the m_frames = " << m_frames;

    const AVCodec *codec = avcodec_find_decoder(formatContext->streams[videoStreamIndex]->codecpar->codec_id);

    if (codecContext)
    {
        avcodec_free_context(&codecContext);
        codecContext = nullptr;
    }

    codecContext = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codecContext, formatContext->streams[videoStreamIndex]->codecpar);
    avcodec_open2(codecContext, codec, nullptr);

    if (swsContext)
    {
        sws_freeContext(swsContext);
        swsContext = nullptr;
    }

   swsContext = sws_getContext(codecContext->width, codecContext->height, codecContext->pix_fmt,
                               codecContext->width, codecContext->height, AV_PIX_FMT_RGB24,
                               SWS_BILINEAR, nullptr, nullptr, nullptr);

qDebug() <<"test FFmepg";
   // 创建目标 AVFrame (BGR)
     AVFrame* bgrFrame = av_frame_alloc();
     bgrFrame->format = AV_PIX_FMT_BGR24;


   while(av_read_frame(formatContext, packet) >= 0) {
        qDebug() <<"test FFmepg";
       if (packet->stream_index == videoStreamIndex)
       {
           qDebug() << "pts = " << packet->pts * av_q2d(formatContext->streams[videoStreamIndex]->time_base);
           qDebug() << "dts = " << packet->dts * av_q2d(formatContext->streams[videoStreamIndex]->time_base);
           qDebug() << "duration = " << packet->duration * av_q2d(formatContext->streams[videoStreamIndex]->time_base);
           avcodec_send_packet(codecContext, packet);
           if (avcodec_receive_frame(codecContext, frame) == 0) {
               bgrFrame->width = frame->width;
               bgrFrame->height = frame->height;
               av_frame_get_buffer(bgrFrame, 0); // 分配内存
               sws_scale(swsContext,
                            frame->data, frame->linesize, 0, frame->height,
                            bgrFrame->data, bgrFrame->linesize);



               cv::Mat mat(bgrFrame->height, bgrFrame->width, CV_8UC3, bgrFrame->data[0], bgrFrame->linesize[0]);
               cv::Mat matCopy = mat.clone();
               cv::cvtColor(matCopy, matCopy, cv::COLOR_BGR2RGB);

               if(ModelInfer)
               {
                   ModelInfer->inference(matCopy);
               }

               emit frameReady(matCopy);
               int delay = static_cast<int>(1000/m_fps);
               QThread::msleep(delay);

//               QImage image(codecContext->width, codecContext->height, QImage::Format_RGB888);
//               uint8_t *data[AV_NUM_DATA_POINTERS] = { image.bits(), nullptr };
//               int linesize[AV_NUM_DATA_POINTERS] = { image.bytesPerLine(), 0 };

//               sws_scale(swsContext, frame->data, frame->linesize, 0, codecContext->height, data, linesize);
//                videoLabel->setPixmap(QPixmap::fromImage(image));
//               QMutexLocker locker(&m_imageMutex);
//                 m_image = image;

//                update();
           }
       }
       av_packet_unref(packet);
   }

    av_frame_free(&bgrFrame);



}
