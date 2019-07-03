#pragma once

#include <QMainWindow>
#include <QListView>
#include "ui_LiveCamWindow.h"
#include <CVImageWidget.h>

#include <EditorWindow.h>
#include <OutputWindow.h>
#include <cvimagewidgetorg.h>
#include <opencv2/opencv.hpp>

#include <QTimer>
#include <QMimeData>
#include <QDrag>
#include <QHash>
#include <QDropEvent>
#include <QMouseEvent>
#include <QEvent>
#include <QFileInfo>
#include <QGraphicsBlurEffect>
#include <QPropertyAnimation>
#include <QSequentialAnimationGroup>
#include <QFile>
#include <QByteArray>
#include <QDebug>
#include <QMessageBox>

#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/range/iterator_range.hpp>

#include <cvimagewidget.h>
#include <QCameraInfo>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <QThread>
#include <FilterUiModel.h>
#include <Tracking/Detector.h>

#include <FilterButton.h>

#include <FilterManager.h>

#define MAX_NUM_CAM 8

QT_BEGIN_NAMESPACE
namespace Ui { class LiveCamWindow; }
QT_END_NAMESPACE

class QThread;
class QLabel;

class LiveCamWindow;

class FilterButton : public QWidget
{
	Q_OBJECT;

public:
	static const QString uncheckedStylesheet;
	static const QString checkedStylesheet;

	FilterButton(const QString buttonName, const QString iconPath, LiveCamWindow* liveCamWindow);

	void setChecked(bool value);
	void setIcon(QIcon icon);

protected:
	LiveCamWindow* liveCamWindow;

	QLabel* iconLabel;
	QLabel* nameLabel;

	void paintEvent(QPaintEvent *) override;
	void mousePressEvent(QMouseEvent *event) override;
};

class LabelShapedButton : public QPushButton
{
	Q_OBJECT;

public:

	static const int verticalPadding;
	static const int horizontalPadding;
	static const QFont font;

	LabelShapedButton(QString text, bool triangleAtRightSide, QColor textColor, QColor topColor, QColor bottomColor, QWidget* parent);

protected:
	void paintEvent(QPaintEvent *event) override;

	QPainterPath path;
	QRegion region;
	QLinearGradient gradient;
	QFontMetrics textMeasurer;
	QColor textColor;
};

class LiveCamWindow : public QMainWindow
{
	Q_OBJECT;

public:
	FilterManager filterManager;

    LiveCamWindow(QWidget *parent = Q_NULLPTR);
	~LiveCamWindow();

	EditorWindow *editorWindow;
	OutputWindow *outputWindow;
	//LabelShapedButton *showFiltersButton;
	//LabelShapedButton *startOutputButton;

	bool event(QEvent *e);

	Ui::LiveCamWindow ui;

	FilterButton *activeFilterButton;

	std::vector<std::string> cameraNames;

public slots:

    void setCompositeLeftWidget(CVImageWidget* imageWidget);
    void on_cameraList_currentIndexChanged(int index);

    void on_showFiltersButton_clicked();
	void on_startOutputButton_clicked();
	void on_maxTargetSlider_valueChanged(int value);
    void closeEvent(QCloseEvent*);
    void on_actionExit_triggered();
    void timerHandler();
	void stopTimer();
	void setRefresh(double fpsLimit);

	void on_create_filter_clicked();
	void on_edit_filter_clicked();
	void on_delete_filter_clicked();
	void on_import_filter_clicked();

	void on_editor_window_closed();
	void on_editor_moduleIcon_changed();
	void on_editor_filterIcon_changed();

	void initCameraAndVideoFileList();
    void initCameraList();
	void initVideoFileList();

	void on_filterButton_clicked(FilterButton* sender);

	void on_moduleListLeft_clicked();
	void on_moduleListRight_clicked();
	void on_trackFaces_stateChanged(int state);
	void on_sameFiltersForAllFaces_stateChanged(int state);
	void on_showFps_stateChanged(int state);

private slots:
	void runTracker2();
    void displayFrame(cv::Mat3b frame, int index);

private:

	friend class EditorWindow;
	friend class OutputWindow;

    CVImageWidget *rawLeftWidget;
    CVImageWidget *rawRightWidget;
    CVImageWidget *compositeLeftWidget;
    CVImageWidget *compositeRighttWidget;
    int numCams;
    QLabel *labels[MAX_NUM_CAM];
   // QThread* threads[MAX_NUM_CAM];
   // CameraWorker* workers[MAX_NUM_CAM];

    QString showFilterString;
    QString hideFilterString;

	QString startOutputString;
	QString stopOutputString;

	bool isDraggingOver;

	QThread* cameraThread;

    bool eventFilter(QObject *obj, QEvent *event);
    void installEventFilters();
	void onCamEventAction(QString camFileName, int index);

	void animatedSubmenuReplacing(std::shared_ptr<FX>);
	void animatedMainMenuReplacing(bool swapToEditor);

	std::vector<FilterUiModel> filterUiModels;

	void setFilter(int value);
	void setFilter(std::shared_ptr<FX> filter);

	void addFilterButton(FilterButton* newbutton);
	QPushButton* addModuleButton();

public:
	QTimer *timer;

	bool isVideoStarted;
	bool areFacesTracked();
	bool isFilterDisplaying;
	bool isSurfaceSelected;
	bool isOutputing;

	CVImageWidgetOrg* outputImageWidget;

	bool addFilter(FilterUiModel& externalInfo);

private:
	void addFilter(bool rewrite);
};
