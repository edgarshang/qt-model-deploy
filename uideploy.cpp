#include "uideploy.h"
#include <QImage>
#include <QDebug>
#include <QString>
#include <QFileDialog>
#include <QMessageBox>
#include <QSettings>

const char *modetye[] = {"YOLOv5", "YOLOv8", "FasterRcnn", "MaskRcnn", "Unet", "resnet18"};

Deploy::Deploy(QWidget *parent)
    : QWidget(parent)
{
    uiInit();
}

void Deploy::uiInit()
{
      uilayout.addWidget(&uileftModelInit());
      uilayout.addWidget(&uiStackWidgetInit());
      uilayout.addLayout(&uiButtonInit());
      uilayout.addStretch(1);
      uilayout.addLayout(&uiShowInit());
//      uilayout.addStretch(1);
      setLayout(&uilayout);

      initSetting();
}

void Deploy::imageshow(cv::Mat &image)
{
    QImage dst(image.data, image.cols, image.rows, static_cast<int>(image.step), QImage::Format::Format_RGB888);
    QshowLabel.setPixmap(QPixmap::fromImage(dst.rgbSwapped()));
}

QWidget& Deploy::uileftModelInit()
{

    for(int i = 0; i < sizeof(modetye)/sizeof(modetye[0]); i++)
    {
        leftModeListWidget.insertItem(i, modetye[i]);
    }

    return leftModeListWidget;
}

QWidget& Deploy::uiStackWidgetInit()
{
    stackWidgetGroup.setTitle("Type");

    QLabel *pathLabel = new QLabel(tr("path:"));
    pathLineEdit = new QLineEdit();

    QHBoxLayout *pathLayout = new QHBoxLayout;
    pathLayout->addWidget(pathLabel);
    pathLayout->addWidget(pathLineEdit);

    QGroupBox *DeployModelTypeGroupBox = new QGroupBox(tr("Deploy ModeType"));
    onnxruntimeRadioBtn = new QRadioButton(tr("&onnxruntime"));
    opvinoRadioBtn      = new QRadioButton(tr("&openvino"));

    QVBoxLayout *vbox = new QVBoxLayout;
    vbox->addWidget(onnxruntimeRadioBtn);
    vbox->addWidget(opvinoRadioBtn);
    vbox->addStretch(1);
    DeployModelTypeGroupBox->setLayout(vbox);

    QVBoxLayout *configLayout = new QVBoxLayout;
    configLayout->addLayout(pathLayout);
    configLayout->addWidget(DeployModelTypeGroupBox);
    configLayout->addStretch(1);
    stackWidgetGroup.setLayout(configLayout);


    onnxruntimeRadioBtn->setChecked(true);

    return stackWidgetGroup;

}


QLayout& Deploy::uiButtonInit()
{
    openfileButton = new QPushButton("Openfile");
    RunButton = new QPushButton("Run");

    buttonLayout.addWidget(openfileButton);
    buttonLayout.addWidget(RunButton);
    buttonLayout.addStretch(1);

    connect(openfileButton, SIGNAL(clicked(bool)), this, SLOT(onPushButtonClick()));
    connect(RunButton, SIGNAL(clicked(bool)), this, SLOT(onPushButtonClick()));

    return buttonLayout;
}

void Deploy::onPushButtonClick()
{
    QPushButton* btn =  (QPushButton*)sender();
    QString text = btn->text();
//    qDebug() << text;
    if( text == "Openfile")
    {
        QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"),
                                                                  "/home",
                                                                  tr("Images (*.png *.jpg *.png *.mp4)"));
//        qDebug() << fileName;
        pathLineEdit->setText(fileName);
        QSettings initSetting("config.ini", QSettings::IniFormat);
        initSetting.setValue("/init/path", fileName);
    }else if( text == "Run" )
    {
        modelTypeInfo.filePath = pathLineEdit->text();
        modelTypeInfo.modelType = modetye[leftModeListWidget.currentRow()];
        modelTypeInfo.deploymode = onnxruntimeRadioBtn->isChecked() ? OnnxRunTime : Openvino;

//        qDebug() << modelTypeInfo.modelType;
//        qDebug() << modelTypeInfo.deploymode;
//        qDebug() << modelTypeInfo.filePath;

        if( modelTypeInfo.filePath != NULL)
        {
            imageProcess->processor(modelTypeInfo);
        }else
        {
            QMessageBox::warning(this, "No ImagePath","Please input the imagePath", QMessageBox::Ok);
        }


    }


}

void Deploy::initSetting()
{
    QSettings initSetting("config.ini", QSettings::IniFormat);
    QString imagePath = initSetting.value("/init/path").toString();
    if( !imagePath.isEmpty())
    {
        pathLineEdit->setText(imagePath);
    }
}

QLayout& Deploy::uiShowInit()
{
    QHBoxLayout *showLayout = new QHBoxLayout;
    QshowLabel.setMinimumSize(480,400);
    QshowLabel.setStyleSheet("QLabel{background-color:rgb(0,0,0);}");
    QshowLabel.setScaledContents(true);
    showLayout->addWidget(&QshowLabel);

    return *showLayout;

}

Deploy::~Deploy()
{
    delete openfileButton;
    delete RunButton;
    delete pathLineEdit;
    delete onnxruntimeRadioBtn;
    delete opvinoRadioBtn;

}
