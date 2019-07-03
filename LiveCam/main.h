#pragma once

#include <opencv2/opencv.hpp>
#include <QPoint.h>
#include <FX.h>

void showComposite(bool visible, std::shared_ptr<FX> filter);

void sendEffectToGraphics(std::shared_ptr<FX> filter);

void closeCameras();
void takeSnapshot(int x, int y);
void runTracker();

bool usingDragDrop(bool value);

bool dropImage(FilterModule *module);

bool doubleClickSelection(FilterModule *module);

bool draggingOver(QPoint qPoint);

void loadFilter(int value);

int startVideoFile(const std::string& fileName);
int startCamera(int cameraNumber);

void setWatermark(std::string watermarkString);

void setSameFiltersForAllFaces(bool value);

void showFps();
void hideFps();
void drawFps(cv::Mat frame);
