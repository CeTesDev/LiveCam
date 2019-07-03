#pragma once

#include <QMainWindow>
#include "ui_OutputWindow.h"
#include <CVImageWidgetOrg.h>
#include <QKeyEvent>
#include <QDebug>

QT_BEGIN_NAMESPACE
namespace Ui { class OutputWindow; }
QT_END_NAMESPACE

class LiveCamWindow;

class OutputWindow : public QMainWindow
{
	Q_OBJECT

public:
	OutputWindow(LiveCamWindow* liveCamWindow, QWidget *parent = Q_NULLPTR);
	~OutputWindow();

	LiveCamWindow* liveCamWindow;

	bool event(QEvent * e);

	public slots:
	void setWidget(CVImageWidgetOrg* widget);
	void on_actionFull_Screen_triggered();

protected:
	void keyPressEvent(QKeyEvent *event); // declaration

private:
	Ui::OutputWindow ui;
	CVImageWidgetOrg *imageWidget;

};

#include <LiveCamWindow.h>