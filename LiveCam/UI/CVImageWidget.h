#pragma once

#include <QObject>

#include <QLabel>
#include <QImage>
#include <QPainter>
#include <QDebug>
#include <opencv2/opencv.hpp>

class CVImageWidget : public QLabel
{
	Q_OBJECT

public:
	CVImageWidget(QObject *parent);
	~CVImageWidget();

    explicit CVImageWidget(QWidget *parent = 0);

    QSize sizeHint() const;
//    QSize minimumSizeHint() const;
public slots :
	void showImage(const cv::Mat& image);
	
	//QPoint showImage(const cv::Mat& image);

protected:
	void paintEvent(QPaintEvent* /*event*/);

	QImage _qimage;
	bool hasImage;
	cv::Mat _tmp;
};
