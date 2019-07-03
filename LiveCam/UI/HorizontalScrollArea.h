#pragma once

#include <QObject>
#include <QScrollArea>
#include <QDebug>

class HorizontalScrollArea : public QScrollArea
{
	Q_OBJECT

public:
	HorizontalScrollArea(QObject *parent);
	~HorizontalScrollArea();

    explicit HorizontalScrollArea(QWidget *parent = 0);

	virtual void wheelEvent(QWheelEvent *ev);
};
