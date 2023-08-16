#ifndef MODELHANDLER_H
#define MODELHANDLER_H

#include <QString>
#include <QThread>
#include "common_api.h"


class ModelHandler : public ImageProcessor
{
public:
    ModelHandler(Show *imageDisplay);
    virtual void processor(modelTypeInfo_ &info);
    Show *display;



};

#endif // MODELHANDLER_H
