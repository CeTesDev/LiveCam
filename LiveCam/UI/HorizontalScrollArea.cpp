#include "HorizontalScrollArea.h"
#include <QWheelEvent>
#include <QStyleOption>
#include <QPainter>

HorizontalScrollArea::~HorizontalScrollArea()
{
}

HorizontalScrollArea::HorizontalScrollArea(QWidget *parent) : QScrollArea(parent)
{

}


void HorizontalScrollArea::wheelEvent(QWheelEvent *ev)
{
	QWheelEvent weHorizontal(ev->pos(), ev->delta(), ev->buttons(), ev->modifiers(), Qt::Horizontal);
	QScrollArea::wheelEvent(&weHorizontal);
}