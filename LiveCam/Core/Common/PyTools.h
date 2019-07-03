#pragma once

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <opencv2/core/core.hpp>
#include <boost/python.hpp>
#include <cstdio>

namespace pytools
{

	static PyObject* opencv_error = 0;

#define ERRWRAP2(expr) \
try \
{ \
    PyAllowThreads allowThreads; \
    expr; \
} \
catch (const cv::Exception &e) \
{ \
    PyErr_SetString(opencv_error, e.what()); \
    return 0; \
}

	static int failmsg(const char *fmt, ...) {
		char str[1000];

		va_list ap;
		va_start(ap, fmt);
		vsnprintf_s(str, sizeof(str), fmt, ap);
		va_end(ap);

		PyErr_SetString(PyExc_TypeError, str);
		return 0;
	}

	class PyAllowThreads {
	public:
		PyAllowThreads() :
			_state(PyEval_SaveThread()) {
		}
		~PyAllowThreads() {
			PyEval_RestoreThread(_state);
		}
	private:
		PyThreadState* _state;
	};

	class PyEnsureGIL {
	public:
		PyEnsureGIL() :
			_state(PyGILState_Ensure()) {
		}
		~PyEnsureGIL() {
			PyGILState_Release(_state);
		}
	private:
		PyGILState_STATE _state;
	};

	class NumpyAllocator :
		public cv::MatAllocator {
	public:
		NumpyAllocator() {
			stdAllocator = cv::Mat::getStdAllocator();
		}
		~NumpyAllocator() {
		}

		cv::UMatData* allocate(PyObject* o, int dims, const int* sizes, int type,
			size_t* step) const {
			cv::UMatData* u = new cv::UMatData(this);
			u->data = u->origdata = (uchar*)PyArray_DATA((PyArrayObject*)o);
			npy_intp* _strides = PyArray_STRIDES((PyArrayObject*)o);
			for (int i = 0; i < dims - 1; i++)
				step[i] = (size_t)_strides[i];
			step[dims - 1] = CV_ELEM_SIZE(type);
			u->size = sizes[0] * step[0];
			u->userdata = o;
			return u;
		}

		cv::UMatData* allocate(int dims0, const int* sizes, int type, void* data,
			size_t* step, cv::AccessFlag flags, cv::UMatUsageFlags usageFlags) const {
			if (data != 0) {
				CV_Error(cv::Error::StsAssert, "The data should normally be NULL!");
				// probably this is safe to do in such extreme case
				return stdAllocator->allocate(dims0, sizes, type, data, step, flags,
					usageFlags);
			}
			PyEnsureGIL gil;

			int depth = CV_MAT_DEPTH(type);
			int cn = CV_MAT_CN(type);
			const int f = (int)(sizeof(size_t) / 8);
			int typenum =
				depth == CV_8U ? NPY_UBYTE :
				depth == CV_8S ? NPY_BYTE :
				depth == CV_16U ? NPY_USHORT :
				depth == CV_16S ? NPY_SHORT :
				depth == CV_32S ? NPY_INT :
				depth == CV_32F ? NPY_FLOAT :
				depth == CV_64F ?
				NPY_DOUBLE :
				f * NPY_ULONGLONG + (f ^ 1) * NPY_UINT;
			int i, dims = dims0;
			cv::AutoBuffer<npy_intp> _sizes(dims + 1);
			for (i = 0; i < dims; i++)
				_sizes[i] = sizes[i];
			if (cn > 1)
				_sizes[dims++] = cn;
			PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
			if (!o)
				CV_Error_(cv::Error::StsError,
				("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
			return allocate(o, dims0, sizes, type, step);
		}

		bool allocate(cv::UMatData* u, cv::AccessFlag accessFlags,
			cv::UMatUsageFlags usageFlags) const {
			return stdAllocator->allocate(u, accessFlags, usageFlags);
		}

		void deallocate(cv::UMatData* u) const {
			if (u) {
				PyEnsureGIL gil;
				PyObject* o = (PyObject*)u->userdata;
				Py_XDECREF(o);
				delete u;
			}
		}

		const MatAllocator* stdAllocator;
	};

	NumpyAllocator g_numpyAllocator;

	PyObject* fromMatToNDArray(const cv::Mat& m) {
		if (!m.data)
			Py_RETURN_NONE;
		cv::Mat temp,
			*p = (cv::Mat*)&m;
		if (!p->u || p->allocator != &g_numpyAllocator) {
			temp.allocator = &g_numpyAllocator;
			ERRWRAP2(m.copyTo(temp));
			p = &temp;
		}
		PyObject* o = (PyObject*)p->u->userdata;
		if (o != nullptr)
		{
			Py_INCREF(o);
		}
		return o;
	}

	cv::Mat fromNDArrayToMat(PyObject* o) {
		cv::Mat m;
		bool allowND = true;
		if (!PyArray_Check(o)) {
			failmsg("argument is not a numpy array");
			if (!m.data)
				m.allocator = &g_numpyAllocator;
		}
		else {
			PyArrayObject* oarr = (PyArrayObject*)o;

			bool needcopy = false, needcast = false;
			int typenum = PyArray_TYPE(oarr), new_typenum = typenum;
			int type = typenum == NPY_UBYTE ? CV_8U : typenum == NPY_BYTE ? CV_8S :
				typenum == NPY_USHORT ? CV_16U :
				typenum == NPY_SHORT ? CV_16S :
				typenum == NPY_INT ? CV_32S :
				typenum == NPY_INT32 ? CV_32S :
				typenum == NPY_FLOAT ? CV_32F :
				typenum == NPY_DOUBLE ? CV_64F : -1;

			if (type < 0) {
				if (typenum == NPY_INT64 || typenum == NPY_UINT64
					|| type == NPY_LONG) {
					needcopy = needcast = true;
					new_typenum = NPY_INT;
					type = CV_32S;
				}
				else {
					failmsg("Argument data type is not supported");
					m.allocator = &g_numpyAllocator;
					return m;
				}
			}

#ifndef CV_MAX_DIM
			const int CV_MAX_DIM = 32;
#endif

			int ndims = PyArray_NDIM(oarr);
			if (ndims >= CV_MAX_DIM) {
				failmsg("Dimensionality of argument is too high");
				if (!m.data)
					m.allocator = &g_numpyAllocator;
				return m;
			}

			int size[CV_MAX_DIM + 1];
			size_t step[CV_MAX_DIM + 1];
			size_t elemsize = CV_ELEM_SIZE1(type);
			const npy_intp* _sizes = PyArray_DIMS(oarr);
			const npy_intp* _strides = PyArray_STRIDES(oarr);
			bool ismultichannel = ndims == 3 && _sizes[2] <= CV_CN_MAX;

			for (int i = ndims - 1; i >= 0 && !needcopy; i--) {
				// these checks handle cases of
				//  a) multi-dimensional (ndims > 2) arrays, as well as simpler 1- and 2-dimensional cases
				//  b) transposed arrays, where _strides[] elements go in non-descending order
				//  c) flipped arrays, where some of _strides[] elements are negative
				if ((i == ndims - 1 && (size_t)_strides[i] != elemsize)
					|| (i < ndims - 1 && _strides[i] < _strides[i + 1]))
					needcopy = true;
			}

			if (ismultichannel && _strides[1] != (npy_intp)elemsize * _sizes[2])
				needcopy = true;

			if (needcopy) {

				if (needcast) {
					o = PyArray_Cast(oarr, new_typenum);
					oarr = (PyArrayObject*)o;
				}
				else {
					oarr = PyArray_GETCONTIGUOUS(oarr);
					o = (PyObject*)oarr;
				}

				_strides = PyArray_STRIDES(oarr);
			}

			for (int i = 0; i < ndims; i++) {
				size[i] = (int)_sizes[i];
				step[i] = (size_t)_strides[i];
			}

			// handle degenerate case
			if (ndims == 0) {
				size[ndims] = 1;
				step[ndims] = elemsize;
				ndims++;
			}

			if (ismultichannel) {
				ndims--;
				type |= CV_MAKETYPE(0, size[2]);
			}

			if (ndims > 2 && !allowND) {
				failmsg("%s has more than 2 dimensions");
			}
			else {

				m = cv::Mat(ndims, size, type, PyArray_DATA(oarr), step);
				m.u = g_numpyAllocator.allocate(o, ndims, size, type, step);
				m.addref();

				if (!needcopy) {
					Py_INCREF(o);
				}
			}
			m.allocator = &g_numpyAllocator;
		}
		return m;
	}


	std::array<cv::Point2f, 4> fromNDArrayToPoints(PyObject* o) {
		std::array<cv::Point2f, 4> result;
		result.fill(cv::Point2f(0,0));
		bool allowND = true;
		if (!PyArray_Check(o)) {
			failmsg("argument is not a numpy array");
		}
		else {
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
				failmsg("%s has more than 2 dimensions");
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
}
