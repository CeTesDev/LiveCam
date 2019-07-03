#pragma comment(linker, "/SUBSYSTEM:windows /ENTRY:mainCRTStartup")

#include "main.h"

#include <math.h>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <QStyleFactory>
#include <QtWidgets/QApplication>

#include <GLSL/GLSL.h>
#include <Common/sceneSwitchDetector.h>
#include <Common/Resolutions.h>
#include <CVImageWidget.h>
#include <CVImageWidgetOrg.h>
#include <LiveCamWindow.h>
#include <FX.h>
#include <Tracking/Detector.h>
#include <Graphics/GraphicsMain.h>
#include <FrameManager.h>
#include <ResourceManager.h>

#ifdef LC_OUTPUT_VIDEO

cv::VideoWriter outputVideo;
#endif

std::string lastVideoFileName;
int videoFrameCounter = 0;
cv::VideoCapture inputVideo;

const static int CAMERA_CLOSED = -1;
int leftViewerFeedIndex;

bool isUsingDragAndDrop = false;
bool dropped = false;
bool isShowingBoundingBox = false;
int currentChosenFace = 0;
QString updatedImageDrop;

int selectedCameraIndex = CAMERA_CLOSED;
cv::VideoCapture camera0;

CVImageWidget* compositeImageWidget;
CVImageWidgetOrg* outputImageWidget;
Detector detector;
GraphicsMain renderer;
FrameManager frameManager;

ShaderManagerWrapper shaderManagerWrapper;

ResourceManager resourceManager;

cv::Mat teamLogo;
cv::Mat filterWatermarkImage;
std::string brandingImage = "";

double fpsLimit = 60.0;
auto lastTimeStamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

//Temp workaround to destroy shader manager in time
class ShaderManagerDestroyer
{
public:
	~ShaderManagerDestroyer()
	{
		shaderManagerWrapper.Clear();
	}
};

cv::Mat imageBoundingBox;

const int amountOfImagesToUse = 7;
cv::Mat imageSequence[amountOfImagesToUse];
int imageSequenceCounter;
bool imageSequenceUp;

QPoint mousePoint;
bool facesForDropping[MAX_TO_TRACK];
float scale = 1.0;

LiveCamWindow *liveCamWindow;

cv::Mat testImage;

bool ShowFps = false;
bool isSameFilterForAllFaces = false;

void sendEffectToGraphics(std::shared_ptr<FX> filter)
{
	//detector.stop();
	/*if (filter != nullptr)
	{
		renderer.loadEffect(nullptr);
	}*/
	renderer.loadEffect(filter.get());
	//detector.start();
}

void loadFilter(int value)
{
	liveCamWindow->filterManager.setCurrentFilterWithID(value);
	sendEffectToGraphics(liveCamWindow->filterManager.getCurrentFilter());
}

bool usingDragDrop(bool value)
{
	isUsingDragAndDrop = value;
	return value;
}

void showComposite(bool visible, std::shared_ptr<FX> filter)
{
	sendEffectToGraphics(filter);
}

void openOutputWindow()
{
}

void closeCameras()
{
	if (camera0.isOpened())
	{
		camera0.release();
	}
	renderer.close();
	cv::destroyAllWindows();
	qApp->quit();
}

bool draggingOver(QPoint qPoint)
{
	mousePoint = qPoint;
	return true;
}

bool doubleClickSelection(FilterModule *module)
{
	renderer.updateAllTextures(module);

	return true;
}

bool dropImage(FilterModule *module)
{
	if (isSameFilterForAllFaces)
	{
		renderer.updateAllTextures(module);

		return true;
	}
	else
	{
		if (facesForDropping[currentChosenFace])
		{
			renderer.updateTexture(module, currentChosenFace);

			return true;
		}
	}

	return false;
}

cv::Mat testForDrop(cv::Mat frame, const std::vector<TrackingTarget>& targets)
{
	for (int id = 0; id < targets.size(); id++){

		float mouseX = ((float)mousePoint.x() / (float)compositeImageWidget->width()) * Resolutions::INPUT_ACTUAL_WIDTH;
		float mouseY = ((float)mousePoint.y() / (float)compositeImageWidget->height()) * Resolutions::INPUT_ACTUAL_HEIGHT;

		int faceCenterX = (targets[id].xCenterRaw);
		int faceCenterY = (targets[id].yCenterRaw);
		float faceWidth = targets[id].widthRaw;

		if ((int)mouseX > (int)(faceCenterX - (faceWidth / 2)) && (int)mouseX < (int)(faceCenterX + (faceWidth / 2)))
		{
			if ((int)mouseY >(int)(faceCenterY - (faceWidth / 2)) && (int)mouseY < (int)(faceCenterY + (faceWidth / 2)))
			{
				int roll = targets[id].roll;

				cv::Mat boundingBoxInside = imageBoundingBox;
				float size = faceWidth * ((float)renderer.frame_width / (float)Resolutions::INPUT_ACTUAL_WIDTH);

				boundingBoxInside = CVImageUtil::rotateAndSize(boundingBoxInside, roll, size);

				CVImageUtil::overlayImage(
					frame, 
					boundingBoxInside,
					cv::Point((faceCenterX * ((float)renderer.frame_width / (float)Resolutions::INPUT_ACTUAL_WIDTH)) - (size / 2),
					(faceCenterY * ((float)renderer.frame_height / (float)Resolutions::INPUT_ACTUAL_HEIGHT)) - (size / 2))
				);

				facesForDropping[id] = true;
				currentChosenFace = id;
			}
		}
		else
		{
			facesForDropping[id] = false;
		}
	}
	return frame;
}

void takeSnapshot(int x, int y)
{
	cv::Mat snapShot;
	if (selectedCameraIndex != CAMERA_CLOSED)
	{
		camera0 >> snapShot;
	}
	else
	{
		if (inputVideo.isOpened())
		{
			inputVideo >> snapShot;
		}
	}
	detector.setSnapshot(snapShot, x, y);
}

void runTracker()
{
	if (liveCamWindow->isVideoStarted)
	{
		cv::Mat image0, output0;
		cv::Mat frame_gray, frameSmall;

		if (selectedCameraIndex != CAMERA_CLOSED)
		{
			camera0 >> image0;
		}
		else
		{
			if (inputVideo.isOpened())
			{
				inputVideo >> image0;

				videoFrameCounter++;

				if (videoFrameCounter >= inputVideo.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_COUNT))
				{
					videoFrameCounter = 0;
					inputVideo.set(cv::VideoCaptureProperties::CAP_PROP_POS_FRAMES, 0);
				}
			}
		}

		if (image0.empty())
		{
#ifdef LC_OUTPUT_VIDEO
			if (outputVideo.isOpened())
			{
				outputVideo = cv::VideoWriter();
			}
#endif
			return;
		}


		if (!liveCamWindow->areFacesTracked())
		{
			if (ShowFps)
			{
				drawFps(image0);
			}

			compositeImageWidget->showImage(image0);

			if (liveCamWindow->isOutputing)
			{
				cv::Size size(Resolutions::OUTPUT_WIDTH, Resolutions::OUTPUT_HEIGHT);//the dst image size,e.g.100x100
				cv::Mat dst;//dst image
				resize(image0, dst, size);//resize image
				outputImageWidget->showImage(dst);
			}
		}
		else
		{
			cv::Mat outputImage;
			std::vector<TrackingTarget> targets;

			if (!image0.empty())
			{
				if (detector.activeTrackingType == TRACKER_TYPE::SURFACE && !liveCamWindow->isSurfaceSelected)
				{
					outputImage = image0;
				}
				else
				{
					targets = detector.getData(image0);
					outputImage = renderer.addTrackerValues(image0, targets);
				}
			}

			if (brandingImage != "" || !filterWatermarkImage.empty()) {
				if (!filterWatermarkImage.empty())
				{
					CVImageUtil::overlayImage2(outputImage, filterWatermarkImage, 40, Resolutions::OUTPUT_HEIGHT - filterWatermarkImage.rows - 40);
				}
				else if (!teamLogo.empty())
				{
					CVImageUtil::overlayImage2(outputImage, teamLogo, 40, Resolutions::OUTPUT_HEIGHT - teamLogo.rows - 40);
				}
			}

			if (liveCamWindow->isOutputing)
			{
				cv::Size size(Resolutions::OUTPUT_WIDTH, Resolutions::OUTPUT_HEIGHT);//the dst image size,e.g.100x100
				cv::Mat dst;//dst image

				if (liveCamWindow->isFilterDisplaying)
				{
					resize(outputImage, dst, size);//resize image
					outputImageWidget->showImage(dst);
				}
				else
				{
					resize(image0, dst, size);//resize image
					outputImageWidget->showImage(dst);
				}
			}

			if (isUsingDragAndDrop)
			{
				testForDrop(outputImage, targets);
			}

			if (ShowFps)
			{
				drawFps(outputImage);
			}

			compositeImageWidget->showImage(outputImage);

#ifdef LC_OUTPUT_VIDEO
			outputVideo << outputImage;
#endif

		}
	}
}

int startVideoFile(const std::string& fileName)
{
	selectedCameraIndex = CAMERA_CLOSED;
	if (camera0.isOpened()) {
		camera0.release();
	}

	inputVideo = cv::VideoCapture("./videos/" + fileName);


	if (!inputVideo.isOpened())
	{
		return -1;
	}

	lastVideoFileName = fileName;

	videoFrameCounter = 0;

	Resolutions::INPUT_ACTUAL_WIDTH = inputVideo.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH);
	Resolutions::INPUT_ACTUAL_HEIGHT = inputVideo.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT);
	if (fpsLimit != 60.0)
	{
		inputVideo.set(cv::VideoCaptureProperties::CAP_PROP_FPS, fpsLimit);
	}

	for (auto filter : liveCamWindow->filterManager.filters)
	{
		filter.second->onInputFrameResized();
	}

	liveCamWindow->editorWindow->getEditableFilter()->onInputFrameResized();

	detector.reset();

	return 0;
}

int startCamera(int cameraNumber)
{
	if (inputVideo.isOpened())
	{
		inputVideo = cv::VideoCapture();
	}


	if (selectedCameraIndex == CAMERA_CLOSED) {

		camera0.open(cameraNumber);

		if (!camera0.isOpened())
		{
			return -1;
		}

		selectedCameraIndex = cameraNumber;
	}
	else if (selectedCameraIndex != cameraNumber) {

		selectedCameraIndex = CAMERA_CLOSED;
		if (camera0.isOpened()) {
			camera0.release();
		}
		camera0.open(cameraNumber);
		if (!camera0.isOpened()) {
			return -1;
		}
		selectedCameraIndex = cameraNumber;
	}

	if (camera0.isOpened()) {
		camera0.set(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH, Resolutions::INPUT_WIDTH);
		camera0.set(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT, Resolutions::INPUT_HEIGHT);
		if (fpsLimit != 60.0)
		{
			camera0.set(cv::VideoCaptureProperties::CAP_PROP_FPS, fpsLimit);
		}

		Resolutions::INPUT_ACTUAL_WIDTH = camera0.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH);
		Resolutions::INPUT_ACTUAL_HEIGHT = camera0.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT);


		for (auto filter : liveCamWindow->filterManager.filters)
		{
			filter.second->onInputFrameResized();
		}

		liveCamWindow->editorWindow->getEditableFilter()->onInputFrameResized();
	}

	leftViewerFeedIndex = cameraNumber;

	detector.reset();

	return 0;
}

void setWatermark(std::string watermarkString)
{
	if (!watermarkString.empty()) {
		filterWatermarkImage = cv::imread(watermarkString, cv::IMREAD_UNCHANGED);
	}
	else
	{
		filterWatermarkImage.release();
	}
}

void setSameFiltersForAllFaces(bool value)
{
	FX* currentEffect = liveCamWindow->filterManager.getCurrentFilter().get();

	isSameFilterForAllFaces = value;

	if (value)
	{
		renderer.updateAllTextures(&currentEffect->filterModules[currentEffect->initialModules[0]]);
	}
	else
	{
		for (int i = 0; i < MAX_TO_TRACK; ++i)
		{
			currentEffect->initialModulesRandomize
				? renderer.updateTexture(&currentEffect->filterModules[rand() % currentEffect->filterModules.size()], i)
				: renderer.updateTexture(&currentEffect->filterModules[currentEffect->initialModules[i]], i);
		}
	}
}

void showFps()
{
	ShowFps = true;
}

void hideFps()
{
	ShowFps = false;
}

void drawFps(cv::Mat frame)
{
	auto nowTimeStamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

	if (lastTimeStamp != nowTimeStamp)
	{

		double fps = 1.0 / ((nowTimeStamp.count() - lastTimeStamp.count()) * 0.001);
		int fpsInt = static_cast<int>(fps);

		std::string limitString = (fpsLimit != 60.0) ? "Max FPS: " + boost::lexical_cast<std::string>(boost::format("%.2f") % fpsLimit) + " / Actual " : "";
		cv::putText(frame,
			(limitString + "FPS: " + boost::lexical_cast<std::string>(fpsInt)).c_str(),
			cv::Point(5, 30),
			cv::FONT_HERSHEY_PLAIN,
			2.0,
			cv::Scalar(255, 0, 0),
			1,
			cv::LineTypes::LINE_AA);

	}

	lastTimeStamp = nowTimeStamp;
}


int main(int argc, char** argv) {

	QApplication app(argc, argv);

	AntiAliasing antiAliasing = AntiAliasing::NONE;
	std::string antiAliasingString = "NONE";

	std::string faceDetector;
	std::string landmarkDetector;
	int maxTargets = 5;
	int targetLocatedDelay = 500;
	int targetLostDelay = 500;
	double scanIntensity = 1.0;

	try
	{
		boost::property_tree::ptree config;
		boost::property_tree::read_json("./assets/application_config.json", config);

		Resolutions::INPUT_WIDTH = config.get<int>("inputWidth", Resolutions::INPUT_WIDTH);
		Resolutions::INPUT_HEIGHT = config.get<int>("inputHeight", Resolutions::INPUT_HEIGHT);

		Resolutions::OUTPUT_WIDTH = config.get<int>("outputWidth", Resolutions::OUTPUT_WIDTH);
		Resolutions::OUTPUT_HEIGHT = config.get<int>("outputHeight", Resolutions::OUTPUT_HEIGHT);

		antiAliasingString = config.get<std::string>("antiAliasing", "NONE"); // "NONE", "SSAA_4X" 
		faceDetector = config.get<std::string>("faceDetector", ""); // Caffe", "Tensorflow", "Yolo"
		landmarkDetector = config.get<std::string>("landmarkDetector", ""); // "Caffe", "Caffe3D" 

		maxTargets = config.get<int>("maxTargets", 5);
		targetLocatedDelay = config.get<int>("targetLocatedDelay", 500);
		targetLostDelay = config.get<int>("targetLostDelay", 200);

		ValueSmoother::SMOOTH_MODIFIER_XY = config.get<int>("xySmoothing", 30);
		ValueSmoother::SMOOTH_MODIFIER_Z = config.get<int>("zSmoothing", 60);

		brandingImage = config.get<std::string>("brandingImage", "");
		if (brandingImage != "") {
			teamLogo = cv::imread(brandingImage, cv::IMREAD_UNCHANGED);
		}

		fpsLimit = config.get<double>("fpsLock", 60.0);
		scanIntensity = config.get<double>("scanIntensity", 1.0); // between 1.0 and 20.0
	}
	catch (...)
	{
		qDebug() << "Couldn't load app config, default resolution settings";
	}

	try
	{
		antiAliasing = AntiAliasingMap.at(antiAliasingString);
	}
	catch (...)
	{
		qDebug() << "Wrong anti-aliasing setting, none anti-aliasing will be applied.";
	};

	// initialize the opengl window for processing
	renderer.init(antiAliasing);

	GLint p = 0;
	glGetIntegerv(GL_MAX_TEXTURE_SIZE, &p);
	qDebug() << p;
	p = 0;

	detector.init(faceDetector, landmarkDetector, targetLocatedDelay, targetLostDelay, scanIntensity);

	// Create the outputImageWidget widget and set it in the UI
	outputImageWidget = new CVImageWidgetOrg();

	ShaderManagerDestroyer shaderManagerDestroyer; //Must be declared before LiveCamWindow

	LiveCamWindow liveWindow; // Create the Main Window
	liveCamWindow = &liveWindow;
	if (fpsLimit != 60)
	{
		liveCamWindow->setRefresh(fpsLimit);
	}
	liveCamWindow->show();

	liveCamWindow->ui.maxTargetSlider->setValue(maxTargets);

	liveCamWindow->outputWindow->setWidget(outputImageWidget);

	//liveCamWindow->outputWindow->setWindowFlags(Qt::CustomizeWindowHint);

	liveCamWindow->outputWindow->setObjectName("outputWindow");
	liveCamWindow->outputWindow->setStyleSheet("#outputWindow { border: solid black; }");
	liveCamWindow->outputWindow->setStyleSheet("background-color: black;");

	// Create the rawLeftImageWidget widget and set it in the UI
	//rawImageWidget = new CVImageWidget();
	//liveCamWindow->setRawLeftWidget(rawImageWidget);

	// Create the image widget
	compositeImageWidget = new CVImageWidget();
	liveCamWindow->setCompositeLeftWidget(compositeImageWidget);

	// Set camera defaults
	leftViewerFeedIndex = CAMERA_CLOSED;

	imageBoundingBox = cv::imread("./assets/images/bounding_box2.png", cv::IMREAD_UNCHANGED);

#ifdef LC_SOURCE_VIDEO

	if (!inputVideo.isOpened())
	{
		cout << "Could not open the input video" << endl;
		return -1;
	}
	int ex = static_cast<int>(inputVideo.get(CV_CAP_PROP_FOURCC));

	char EXT[] = { (char)(ex & 0XFF) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0 };

	cv::Size S = cv::Size((int)inputVideo.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH),    // Acquire input size
		(int)inputVideo.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT));

#endif
#ifdef LC_OUTPUT_VIDEO

	auto nowTime = std::chrono::system_clock::now();

	auto in_time_t = std::chrono::system_clock::to_time_t(nowTime);

	std::stringstream ss;
	ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H.%M.%S");

	//outputVideo.open("output"+ ss.str()+".avi", ex, inputVideo.get(CV_CAP_PROP_FPS), S, true);
	outputVideo.open("output" + ss.str() + ".avi", CV_FOURCC('M', 'P', '4', '2'), 16, cv::Size(1920, 1080), true);

	if (!outputVideo.isOpened())
	{
		cout << "Could not open the output video for write" << endl;
		return -1;
	}

	#endif

	detector.start();

	return app.exec();
}
