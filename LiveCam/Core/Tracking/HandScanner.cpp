#include "HandScanner.h"

#ifdef LC_INCLUDE_HAND_TRACKING

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <opencv2/core/types_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>
#include <boost/filesystem.hpp>
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <boost/python.hpp>
#include <Common/CommonClasses.h>

PyObject *HDPyModule, *HDPyDict, *HDPyFunc;

#if PY_MAJOR_VERSION >= 3 
int
#else 
void
#endif 
init_numpy()
{
	import_array();
}

std::array<cv::Point2f, 4> fromNDArrayToPoints(PyObject* o) {
	std::array<cv::Point2f, 4> result;
	result.fill(cv::Point2f(0, 0));
	bool allowND = true;
	if (PyArray_Check(o)) {
		PyArrayObject* oarr = (PyArrayObject*)o;

		float* m = new float[8];
		bool needcopy = false, needcast = false;
		int typenum = NPY_FLOAT;
		int ndims = PyArray_NDIM(oarr);
		const npy_intp* _sizes = PyArray_DIMS(oarr);
		const npy_intp* _strides = PyArray_STRIDES(oarr);

		for (int i = ndims - 1; i >= 0 && !needcopy; i--) {
			// these checks handle cases of
			//  a) multi-dimensional (ndims > 2) arrays, as well as simpler 1- and 2-dimensional cases
			//  b) transposed arrays, where _strides[] elements go in non-descending order
			//  c) flipped arrays, where some of _strides[] elements are negative
			if ((i == ndims - 1 && (size_t)_strides[i] != 9)
				|| (i < ndims - 1 && _strides[i] < _strides[i + 1]))
				needcopy = true;
		}

		if (needcopy) {

			if (needcast) {
				o = PyArray_Cast(oarr, NPY_FLOAT);
				oarr = (PyArrayObject*)o;
			}
			else {
				oarr = PyArray_GETCONTIGUOUS(oarr);
				o = (PyObject*)oarr;
			}

			_strides = PyArray_STRIDES(oarr);
		}

		// handle degenerate case
		if (ndims == 0) {
			ndims++;
		}

		if (ndims > 2 && !allowND) {
			// ERR
		}
		else {
			std::memcpy(m, PyArray_DATA(oarr), sizeof(float) * 8);
			/*float xS = Resolutions::OUTPUT_WIDTH / 300;
			float yS = Resolutions::OUTPUT_HEIGHT / 200;
			result[0] = cv::Point2f((m[1] * xS) * Resolutions::OUTPUT_WIDTH, (m[0] * yS) * Resolutions::OUTPUT_HEIGHT);
			result[1] = cv::Point2f((m[3] * xS) * Resolutions::OUTPUT_WIDTH, (m[2] * yS) * Resolutions::OUTPUT_HEIGHT);
			result[2] = cv::Point2f((m[5] * xS) * Resolutions::OUTPUT_WIDTH, (m[4] * yS) * Resolutions::OUTPUT_HEIGHT);
			result[3] = cv::Point2f((m[7] * xS) * Resolutions::OUTPUT_WIDTH, (m[6] * yS) * Resolutions::OUTPUT_HEIGHT);*/
			if (m[2] < 0.98)
			{
				result[0] = cv::Point2f(m[1] * Resolutions::OUTPUT_WIDTH, m[0] * Resolutions::OUTPUT_HEIGHT);
				result[1] = cv::Point2f(m[3] * Resolutions::OUTPUT_WIDTH, m[2] * Resolutions::OUTPUT_HEIGHT);
			}
			if (m[6] < 0.98)
			{
				result[2] = cv::Point2f(m[5] * Resolutions::OUTPUT_WIDTH, m[4] * Resolutions::OUTPUT_HEIGHT);
				result[3] = cv::Point2f(m[7] * Resolutions::OUTPUT_WIDTH, m[6] * Resolutions::OUTPUT_HEIGHT);
			}

			if (!needcopy) {
				Py_INCREF(o);
			}
			delete[] m;
		}
	}
	return result;
}

HandScanner::HandScanner()
{
	capacity = MAX_TARGETS_COUNT;
	inputSize = cv::Size(200, 200);
	inputMean = cv::Scalar(0, 0, 0);
	inputSwapRB = true;
	detectedPoints.fill(cv::Point2f(0, 0));

	available = true;

#if !defined(_DEBUG)
	Py_SetPath(L".\\python36.zip");
#else
	Py_SetPath(L"C:\\Python\\3.6.8\\Lib");
#endif
	Py_SetProgramName(L"LiveCam");
	Py_InitializeEx(0);
	init_numpy();
	PyRun_SimpleString("import sys\n");
	PyRun_SimpleString("sys.path.append('./')\n");
	PyRun_SimpleString("import detect\n");
	PyObject *pName = PyUnicode_DecodeFSDefault("detect");
	HDPyModule = PyImport_Import(pName);
	if (HDPyModule != nullptr)
	{
		HDPyDict = PyModule_GetDict(HDPyModule);
		HDPyFunc = PyDict_GetItemString(HDPyDict, "get_hands");
	}
	Py_DECREF(pName);
}

HandScanner::~HandScanner()
{
	available = false;
	if (HDPyModule != nullptr)
	{
		Py_XDECREF(HDPyModule);
	}
	Py_XDECREF(HDPyDict);
	Py_XDECREF(HDPyFunc);
	Py_Finalize();
}

void HandScanner::start()
{
}

void HandScanner::stop()
{
}

void HandScanner::setRefFrame(cv::Mat frame, int x, int y)
{}

cv::Mat HandScanner::prepScanFrame(cv::Mat frame)
{
	if (!frame.empty())
	{
		cv::Mat scanFrame;
		cv::Size preSize(300, 178);
		cv::resize(frame, scanFrame, preSize);
		return scanFrame;
	}
	return frame;
}

cv::Mat HandScanner::prepKeypointsFrame(cv::Mat frame)
{
	return frame;
}

std::vector<cv::Rect> HandScanner::scan(cv::Mat frame)
{
	std::vector<cv::Rect> detectedHands;

	scanCounter++;
	if (available)
	{
		available = false;
		if (HDPyFunc != nullptr && PyCallable_Check(HDPyFunc))
		{
			PyObject *pArgs, *pValue;
			pArgs = PyTuple_New(1);
			npy_intp dimensions[3] = { frame.rows, frame.cols, frame.channels() };
			pValue = PyArray_SimpleNewFromData(frame.dims + 1, (npy_intp*)&dimensions, NPY_UINT8, frame.data);
			PyTuple_SetItem(pArgs, 0, pValue);
			PyObject *pResult = PyObject_CallObject(HDPyFunc, pArgs);
			Py_XDECREF(pArgs);

			if (pResult != nullptr) {
				std::array<cv::Point2f, 4> _points = fromNDArrayToPoints(pResult);
				detectedHands.push_back(cv::Rect(_points[0], _points[1]));
				detectedHands.push_back(cv::Rect(_points[2], _points[3]));
				detectedPoints[0] = _points[0];
				detectedPoints[1] = _points[1];
				detectedPoints[2] = _points[2];
				detectedPoints[3] = _points[3];
				Py_XDECREF(pResult);
				available = true;
			}
			else
			{
				Py_XDECREF(pResult);
			}
		}
	}
	else
	{
		delayCounter++;
		if (delayCounter > 200)
		{
			delayCounter = 0;
			scanCounter = 0;
			available = true;
		}
	}

	return detectedHands;
}

void HandScanner::asyncScan()
{
	
}

void HandScanner::keypoints(TargetHoldingStruct &target, std::array<cv::Point2f, TARGET_DETAIL_MODIFIER>& data)
{
	cv::Rect newRect = target.rect;

	constexpr float cx = 1.3;
	constexpr float cy = 0.85;

	int diffX = newRect.width*cx - newRect.width;
	int diffY = newRect.height*cy - newRect.height;

	newRect.width *= cx;
	newRect.height *= 1.0;
	newRect.x -= diffX * 0.5;
	newRect.y -= diffY;

	target.lastKnownRect = target.rect;
	target.rect = newRect;

	if (newRect.x < 0 || newRect.x + newRect.width >= Resolutions::INPUT_ACTUAL_WIDTH
		|| newRect.y < 0 || newRect.y + newRect.height >= Resolutions::INPUT_ACTUAL_HEIGHT)
	{
		target.visible = false;
		return;
	}

	for (size_t i = 0; i < detectedPoints.size(); i++) {
		cv::Point2f pnt = detectedPoints[i];
		data[i] = pnt;
	}
}

void HandScanner::scale(double newscale)
{
	activescale = newscale;
}

#endif
