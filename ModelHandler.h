#ifndef MODELHANDLER_H
#define MODELHANDLER_H

#include <QString>
#include <QThread>
#include <iostream>
#include <memory>
#include "common_api.h"
#include "ort_tutorial.h"


class ModelHandler : public QThread,  public ImageProcessor
{
public:
    ModelHandler(Show *imageDisplay);
    virtual void processor(modelTypeInfo_ &info);
    Show *display;

//    ort_tutorial *ort_test;
    std::shared_ptr<ort_tutorial> ort_test;



};

#endif // MODELHANDLER_H
