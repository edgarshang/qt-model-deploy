#include "uideploy.h"

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
      uilayout.addLayout(&uiShowInit());
      setLayout(&uilayout);
}

QWidget& Deploy::uileftModelInit()
{
    leftModeListWidget.insertItem(0, "YOLOv5");
    leftModeListWidget.insertItem(1, "YOLOv8");
    leftModeListWidget.insertItem(2, "FasterRcnn");
    leftModeListWidget.insertItem(3, "MaskRcnn");
    leftModeListWidget.insertItem(4, "Unet");

    return leftModeListWidget;
}

QWidget& Deploy::uiStackWidgetInit()
{
    stackWidgetGroup.setTitle("Type");

    QLabel *pathLabel = new QLabel(tr("path:"));
    QLineEdit *pathLineEdit = new QLineEdit();

    QHBoxLayout *pathLayout = new QHBoxLayout;
    pathLayout->addWidget(pathLabel);
    pathLayout->addWidget(pathLineEdit);

    QVBoxLayout *configLayout = new QVBoxLayout;
    configLayout->addLayout(pathLayout);
    stackWidgetGroup.setLayout(configLayout);


    return stackWidgetGroup;

}


QLayout& Deploy::uiButtonInit()
{
    openfileButton = new QPushButton("Openfile");
    RunButton = new QPushButton("Run");
    ImageFolderButton = new QPushButton("Folder");

    buttonLayout.addWidget(openfileButton);
    buttonLayout.addWidget(RunButton);
    buttonLayout.addWidget(ImageFolderButton);

    return buttonLayout;
}

QLayout& Deploy::uiShowInit()
{
    QHBoxLayout *showLayout = new QHBoxLayout;
    showLayout->addWidget(&QshowLabel);

    return *showLayout;

}

Deploy::~Deploy()
{
//    delete openfileButton;
//    delete RunButton;
//    delete ImageFolderButton;
//    delete openfileButton;
//    delete RunButton;
//    delete ImageFolderButton;
}
