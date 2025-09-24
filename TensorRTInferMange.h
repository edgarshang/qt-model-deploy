#ifndef TENSORRTINFERMANGE_H
#define TENSORRTINFERMANGE_H

#include "uideploy.h"
#include "ModelHandler.h"

#include <QObject>

class TensorRTInferMange:public QObject
{
    Q_OBJECT
public:
    TensorRTInferMange();
    Deploy m_ui;
    ModelHandler *modelHandle;
};

#endif // TENSORRTINFERMANGE_H
