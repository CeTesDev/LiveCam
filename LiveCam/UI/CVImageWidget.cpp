#include "CVImageWidget.h"
#include <opencv2/opencv.hpp>

#include <qstyleoption.h>

const int BORDER_WIDTH = 4;

CVImageWidget::~CVImageWidget()
{
}

CVImageWidget::CVImageWidget(QWidget *parent) : QLabel(parent)
{
	hasImage = false;
}

QSize CVImageWidget::sizeHint() const
{
    return _qimage.size();
}


void CVImageWidget::showImage(const cv::Mat &image)
{
	hasImage = !image.empty();

	if (image.rows > 0 && image.cols > 0)
	{
		// Convert the image to the RGB888 format
		switch (image.type()) {
		case CV_8UC1:
			cvtColor(image, _tmp, cv::COLOR_GRAY2RGB);
			break;
		case CV_8UC3:
			cvtColor(image, _tmp, cv::COLOR_BGR2RGB);
			break;
		}

		// QImage needs the data to be stored continuously in memory
		assert(_tmp.isContinuous());
		// Assign OpenCV's image buffer to the QImage. Note that the bytesPerLine parameter
		// (http://qt-project.org/doc/qt-4.8/qimage.html#QImage-6) is 3*width because each pixel
		// has three bytes.

		QImage qimage = QImage(_tmp.data, _tmp.cols, _tmp.rows, _tmp.cols * 3, QImage::Format_RGB888);

		auto parent = parentWidget();
		QSize newSize;
		newSize.setWidth(qimage.width() < parent->width() - 60 ? qimage.width() : parent->width() - 60);
		newSize.setHeight(qimage.height() < parent->height() - 60 ? qimage.height() : parent->height() - 60);

		//qDebug() << "CVImageWidgetOrg::showImage :::: parent" << parent_widget->width() << parent_widget->height();
		//  qDebug() << "CVImageWidgetOrg::showImage :::: image " << qimage.width() << qimage.height();
		// qDebug() << "CVImageWidgetOrg::showImage :::: image_" << _qimage.width() << _qimage.height() << "\n\n";

		if (qimage.width() > width() || qimage.height() > height())
		{
			_qimage = qimage.scaled(newSize, Qt::KeepAspectRatio);
		}
		else
		{
			_qimage = qimage;
		}

		setFixedSize(_qimage.size());

		repaint();
	}
}


void CVImageWidget::paintEvent(QPaintEvent* event)
{   	
	// Display the image

	//QStyleOption opt;
	//opt.init(this);
	//style()->drawPrimitive(QStyle::PE_Widget, &opt, &painter, this);

	if (!hasImage)
	{
		QLabel::paintEvent(event);
	}

	QPainter painter(this);

	if (hasImage)
	{
		QRect image_rect(_qimage.rect());
		image_rect.moveCenter(rect().center());
		painter.drawImage(image_rect.topLeft(), _qimage);
	}

	const QRect topBorder = { 0, 0, width(), BORDER_WIDTH };
	const QRect bottomBorder = { 0, height() - 1 - BORDER_WIDTH, width(), height() };
	const QRect leftBorder = { 0, BORDER_WIDTH, BORDER_WIDTH, height() - BORDER_WIDTH };
	const QRect rightBorder = { width() - 1 - BORDER_WIDTH, BORDER_WIDTH, width(), height() - BORDER_WIDTH };
	painter.fillRect(topBorder, QColor(36, 46, 104));
	painter.fillRect(bottomBorder, QColor(36, 46, 104));
	painter.fillRect(leftBorder, QColor(36, 46, 104));
	painter.fillRect(rightBorder, QColor(36, 46, 104));
}
