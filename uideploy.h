#ifndef DEPLOY_H
#define DEPLOY_H

#include <QPushButton>
#include <QListWidget>
#include <QStackedWidget>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QTabWidget>
#include <QGroupBox>
#include <QFormLayout>
#include <QLabel>
#include <QLineEdit>
#include <QRadioButton>

#include <QWidget>

#include "common_api.h"

class Deploy : public QWidget, public Show
{
    Q_OBJECT

public:
    Deploy(QWidget *parent = 0);
    ~Deploy();

public:
    virtual void imageshow(cv::Mat &image);


public:
    void uiInit();
    QWidget& uileftModelInit();
    QWidget& uiStackWidgetInit();
    QLayout& uiButtonInit();
    QLayout& uiShowInit();

public: // button
    QPushButton* openfileButton;
    QPushButton* RunButton;

    QLineEdit *pathLineEdit;
    QRadioButton *onnxruntimeRadioBtn;
    QRadioButton *opvinoRadioBtn;

    modelTypeInfo_ modelTypeInfo;

public slots:
    void onPushButtonClick();
public:
    QListWidget leftModeListWidget;

public:
    QStackedWidget stackWidget;
    QHBoxLayout stackWidgetLayout;
    QGroupBox stackWidgetGroup;

    QWidget settingShowQwidget;


public:
    QVBoxLayout buttonLayout;

public:
    QTabWidget showTabWidget;
    QLabel QshowLabel;

public:
    QHBoxLayout uilayout;

public:
    ImageProcessor *imageProcess;
    void setImageProcesser(ImageProcessor *process){imageProcess = process;}

};

#endif // DEPLOY_H
