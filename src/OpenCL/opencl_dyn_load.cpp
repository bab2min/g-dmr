#include "opencl_dyn_load.h"

__clGetPlatformIDs *clGetPlatformIDs;
__clGetPlatformInfo *clGetPlatformInfo;
__clGetDeviceIDs *clGetDeviceIDs;
__clGetDeviceInfo *clGetDeviceInfo;
__clCreateSubDevices *clCreateSubDevices;
__clRetainDevice *clRetainDevice;
__clReleaseDevice *clReleaseDevice;
__clCreateContext *clCreateContext;
__clCreateContextFromType *clCreateContextFromType;
__clRetainContext *clRetainContext;
__clReleaseContext *clReleaseContext;
__clGetContextInfo *clGetContextInfo;
__clRetainCommandQueue *clRetainCommandQueue;
__clReleaseCommandQueue *clReleaseCommandQueue;
__clGetCommandQueueInfo *clGetCommandQueueInfo;
__clCreateBuffer *clCreateBuffer;
__clCreateSubBuffer *clCreateSubBuffer;
__clCreateImage *clCreateImage;
__clRetainMemObject *clRetainMemObject;
__clReleaseMemObject *clReleaseMemObject;
__clGetSupportedImageFormats *clGetSupportedImageFormats;
__clGetMemObjectInfo *clGetMemObjectInfo;
__clGetImageInfo *clGetImageInfo;
__clSetMemObjectDestructorCallback *clSetMemObjectDestructorCallback;
__clRetainSampler *clRetainSampler;
__clReleaseSampler *clReleaseSampler;
__clGetSamplerInfo *clGetSamplerInfo;
__clCreateProgramWithSource *clCreateProgramWithSource;
__clCreateProgramWithBinary *clCreateProgramWithBinary;
__clCreateProgramWithBuiltInKernels *clCreateProgramWithBuiltInKernels;
__clRetainProgram *clRetainProgram;
__clReleaseProgram *clReleaseProgram;
__clBuildProgram *clBuildProgram;
__clCompileProgram *clCompileProgram;
__clLinkProgram *clLinkProgram;
__clUnloadPlatformCompiler *clUnloadPlatformCompiler;
__clGetProgramInfo *clGetProgramInfo;
__clGetProgramBuildInfo *clGetProgramBuildInfo;
__clCreateKernel *clCreateKernel;
__clCreateKernelsInProgram *clCreateKernelsInProgram;
__clRetainKernel *clRetainKernel;
__clReleaseKernel *clReleaseKernel;
__clSetKernelArg *clSetKernelArg;
__clGetKernelInfo *clGetKernelInfo;
__clGetKernelArgInfo *clGetKernelArgInfo;
__clGetKernelWorkGroupInfo *clGetKernelWorkGroupInfo;
__clWaitForEvents *clWaitForEvents;
__clGetEventInfo *clGetEventInfo;
__clCreateUserEvent *clCreateUserEvent;
__clRetainEvent *clRetainEvent;
__clReleaseEvent *clReleaseEvent;
__clSetUserEventStatus *clSetUserEventStatus;
__clSetEventCallback *clSetEventCallback;
__clGetEventProfilingInfo *clGetEventProfilingInfo;
__clFlush *clFlush;
__clFinish *clFinish;
__clEnqueueReadBuffer *clEnqueueReadBuffer;
__clEnqueueReadBufferRect *clEnqueueReadBufferRect;
__clEnqueueWriteBuffer *clEnqueueWriteBuffer;
__clEnqueueWriteBufferRect *clEnqueueWriteBufferRect;
__clEnqueueFillBuffer *clEnqueueFillBuffer;
__clEnqueueCopyBuffer *clEnqueueCopyBuffer;
__clEnqueueCopyBufferRect *clEnqueueCopyBufferRect;
__clEnqueueReadImage *clEnqueueReadImage;
__clEnqueueWriteImage *clEnqueueWriteImage;
__clEnqueueFillImage *clEnqueueFillImage;
__clEnqueueCopyImage *clEnqueueCopyImage;
__clEnqueueCopyImageToBuffer *clEnqueueCopyImageToBuffer;
__clEnqueueCopyBufferToImage *clEnqueueCopyBufferToImage;
__clEnqueueMapBuffer *clEnqueueMapBuffer;
__clEnqueueMapImage *clEnqueueMapImage;
__clEnqueueUnmapMemObject *clEnqueueUnmapMemObject;
__clEnqueueMigrateMemObjects *clEnqueueMigrateMemObjects;
__clEnqueueNDRangeKernel *clEnqueueNDRangeKernel;
__clEnqueueNativeKernel *clEnqueueNativeKernel;
__clEnqueueMarkerWithWaitList *clEnqueueMarkerWithWaitList;
__clEnqueueBarrierWithWaitList *clEnqueueBarrierWithWaitList;
__clGetExtensionFunctionAddressForPlatform *clGetExtensionFunctionAddressForPlatform;
__clCreateImage2D *clCreateImage2D;
__clCreateImage3D *clCreateImage3D;
__clEnqueueMarker *clEnqueueMarker;
__clEnqueueWaitForEvents *clEnqueueWaitForEvents;
__clEnqueueBarrier *clEnqueueBarrier;
__clUnloadCompiler *clUnloadCompiler;
__clGetExtensionFunctionAddress *clGetExtensionFunctionAddress;
__clCreateCommandQueue *clCreateCommandQueue;
__clCreateSampler *clCreateSampler;
__clEnqueueTask *clEnqueueTask;

__clSetDefaultDeviceCommandQueue *clSetDefaultDeviceCommandQueue;
__clGetDeviceAndHostTimer *clGetDeviceAndHostTimer;
__clGetHostTimer *clGetHostTimer;
__clCreateCommandQueueWithProperties *clCreateCommandQueueWithProperties;
__clCreatePipe *clCreatePipe;
__clGetPipeInfo *clGetPipeInfo;
__clSVMAlloc *clSVMAlloc;
__clSVMFree *clSVMFree;
__clCreateSamplerWithProperties *clCreateSamplerWithProperties;
__clCreateProgramWithIL *clCreateProgramWithIL;
__clCloneKernel *clCloneKernel;
__clSetKernelArgSVMPointer *clSetKernelArgSVMPointer;
__clSetKernelExecInfo *clSetKernelExecInfo;
__clGetKernelSubGroupInfo *clGetKernelSubGroupInfo;
__clEnqueueSVMFree *clEnqueueSVMFree;
__clEnqueueSVMMemcpy *clEnqueueSVMMemcpy;
__clEnqueueSVMMemFill *clEnqueueSVMMemFill;
__clEnqueueSVMMap *clEnqueueSVMMap;
__clEnqueueSVMUnmap *clEnqueueSVMUnmap;
__clEnqueueSVMMigrateMem *clEnqueueSVMMigrateMem;

#if defined(_WIN32) || defined(_WIN64)

#include <Windows.h>

#ifdef UNICODE
static LPCWSTR __ClLibName = L"OpenCl.dll";
#else
static LPCSTR __ClLibName = "OpenCl.dll";
#endif

typedef HMODULE CL_LIBRARY;

cl_int CL_LOAD_LIBRARY(CL_LIBRARY *pInstance) {
	*pInstance = LoadLibrary(__ClLibName);
	if (*pInstance == NULL) {
		return CL_DEVICE_NOT_FOUND;
	}
	return CL_SUCCESS;
}

#define GET_PROC(name)                             \
  name = (__##name *)GetProcAddress(ClLib, #name); \
  if (name == NULL) return CL_DEVICE_NOT_AVAILABLE

#elif defined(__unix__) || defined(__APPLE__) || defined(__MACOSX)

#include <dlfcn.h>

#if defined(__APPLE__) || defined(__MACOSX)
static char __ClLibName[] = "/System/Library/Frameworks/OpenCL.framework/OpenCL";
#else
static char __ClLibName[] = "libOpenCL.so";
#endif

typedef void *CL_LIBRARY;

cl_int CL_LOAD_LIBRARY(CL_LIBRARY *pInstance) {
	*pInstance = dlopen("libOpenCL.so", RTLD_NOW);
	if(*pInstance) return CL_SUCCESS;
	*pInstance = dlopen("/System/Library/Frameworks/OpenCL.framework/OpenCL", RTLD_NOW);
	if(*pInstance) return CL_SUCCESS;
	*pInstance = dlopen("/usr/lib/libOpenCL.dylib", RTLD_NOW);
	if(*pInstance) return CL_SUCCESS;
	return CL_DEVICE_NOT_FOUND;
}

#define GET_PROC(name)                            \
  name = (__##name *)(size_t)dlsym(ClLib, #name); \
  if (name == NULL) return CL_DEVICE_NOT_AVAILABLE

#endif

cl_int clLibLoad(int version) {
	CL_LIBRARY ClLib;
	cl_int result;
	result = CL_LOAD_LIBRARY(&ClLib);
	if (result != CL_SUCCESS) {
		return result;
	}

	GET_PROC(clGetPlatformIDs);
	GET_PROC(clGetPlatformInfo);
	GET_PROC(clGetDeviceIDs);
	GET_PROC(clGetDeviceInfo);
	GET_PROC(clCreateSubDevices);
	GET_PROC(clRetainDevice);
	GET_PROC(clReleaseDevice);
	GET_PROC(clCreateContext);
	GET_PROC(clCreateContextFromType);
	GET_PROC(clRetainContext);
	GET_PROC(clReleaseContext);
	GET_PROC(clGetContextInfo);
	GET_PROC(clRetainCommandQueue);
	GET_PROC(clReleaseCommandQueue);
	GET_PROC(clGetCommandQueueInfo);
	GET_PROC(clCreateBuffer);
	GET_PROC(clCreateSubBuffer);
	GET_PROC(clCreateImage);
	GET_PROC(clRetainMemObject);
	GET_PROC(clReleaseMemObject);
	GET_PROC(clGetSupportedImageFormats);
	GET_PROC(clGetMemObjectInfo);
	GET_PROC(clGetImageInfo);
	GET_PROC(clSetMemObjectDestructorCallback);
	GET_PROC(clRetainSampler);
	GET_PROC(clReleaseSampler);
	GET_PROC(clGetSamplerInfo);
	GET_PROC(clCreateProgramWithSource);
	GET_PROC(clCreateProgramWithBinary);
	GET_PROC(clCreateProgramWithBuiltInKernels);
	GET_PROC(clRetainProgram);
	GET_PROC(clReleaseProgram);
	GET_PROC(clBuildProgram);
	GET_PROC(clCompileProgram);
	GET_PROC(clLinkProgram);
	GET_PROC(clUnloadPlatformCompiler);
	GET_PROC(clGetProgramInfo);
	GET_PROC(clGetProgramBuildInfo);
	GET_PROC(clCreateKernel);
	GET_PROC(clCreateKernelsInProgram);
	GET_PROC(clRetainKernel);
	GET_PROC(clReleaseKernel);
	GET_PROC(clSetKernelArg);
	GET_PROC(clGetKernelInfo);
	GET_PROC(clGetKernelArgInfo);
	GET_PROC(clGetKernelWorkGroupInfo);
	GET_PROC(clWaitForEvents);
	GET_PROC(clGetEventInfo);
	GET_PROC(clCreateUserEvent);
	GET_PROC(clRetainEvent);
	GET_PROC(clReleaseEvent);
	GET_PROC(clSetUserEventStatus);
	GET_PROC(clSetEventCallback);
	GET_PROC(clGetEventProfilingInfo);
	GET_PROC(clFlush);
	GET_PROC(clFinish);
	GET_PROC(clEnqueueReadBuffer);
	GET_PROC(clEnqueueReadBufferRect);
	GET_PROC(clEnqueueWriteBuffer);
	GET_PROC(clEnqueueWriteBufferRect);
	GET_PROC(clEnqueueFillBuffer);
	GET_PROC(clEnqueueCopyBuffer);
	GET_PROC(clEnqueueCopyBufferRect);
	GET_PROC(clEnqueueReadImage);
	GET_PROC(clEnqueueWriteImage);
	GET_PROC(clEnqueueFillImage);
	GET_PROC(clEnqueueCopyImage);
	GET_PROC(clEnqueueCopyImageToBuffer);
	GET_PROC(clEnqueueCopyBufferToImage);
	GET_PROC(clEnqueueMapBuffer);
	GET_PROC(clEnqueueMapImage);
	GET_PROC(clEnqueueUnmapMemObject);
	GET_PROC(clEnqueueMigrateMemObjects);
	GET_PROC(clEnqueueNDRangeKernel);
	GET_PROC(clEnqueueNativeKernel);
	GET_PROC(clEnqueueMarkerWithWaitList);
	GET_PROC(clEnqueueBarrierWithWaitList);
	GET_PROC(clGetExtensionFunctionAddressForPlatform);
	GET_PROC(clCreateImage2D);
	GET_PROC(clCreateImage3D);
	GET_PROC(clEnqueueMarker);
	GET_PROC(clEnqueueWaitForEvents);
	GET_PROC(clEnqueueBarrier);
	GET_PROC(clUnloadCompiler);
	GET_PROC(clGetExtensionFunctionAddress);
	GET_PROC(clCreateCommandQueue);
	GET_PROC(clCreateSampler);
	GET_PROC(clEnqueueTask);

	if (version >= 20)
	{
		GET_PROC(clCreateCommandQueueWithProperties);
		GET_PROC(clCreatePipe);
		GET_PROC(clGetPipeInfo);
		GET_PROC(clSVMAlloc);
		GET_PROC(clSVMFree);
		GET_PROC(clCreateSamplerWithProperties);
		GET_PROC(clSetKernelArgSVMPointer);
		GET_PROC(clSetKernelExecInfo);
		GET_PROC(clEnqueueSVMFree);
		GET_PROC(clEnqueueSVMMemcpy);
		GET_PROC(clEnqueueSVMMemFill);
		GET_PROC(clEnqueueSVMMap);
		GET_PROC(clEnqueueSVMUnmap);
	}

	if (version >= 21)
	{
		GET_PROC(clSetDefaultDeviceCommandQueue);
		GET_PROC(clGetDeviceAndHostTimer);
		GET_PROC(clGetHostTimer);
		GET_PROC(clCreateProgramWithIL);
		GET_PROC(clCloneKernel);
		GET_PROC(clGetKernelSubGroupInfo);
		GET_PROC(clEnqueueSVMMigrateMem);
	}
	return CL_SUCCESS;
}