#include <iostream>
#include <sstream>
#include <fstream>

#include "OpenCLMgr.h"

using namespace std;
using namespace tomoto::ocl;

const char* OpenCLMgr::translateOpenCLError(cl_int errorCode)
{
	switch (errorCode)
	{
	case CL_SUCCESS:                            return "CL_SUCCESS";
	case CL_DEVICE_NOT_FOUND:                   return "CL_DEVICE_NOT_FOUND";
	case CL_DEVICE_NOT_AVAILABLE:               return "CL_DEVICE_NOT_AVAILABLE";
	case CL_COMPILER_NOT_AVAILABLE:             return "CL_COMPILER_NOT_AVAILABLE";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case CL_OUT_OF_RESOURCES:                   return "CL_OUT_OF_RESOURCES";
	case CL_OUT_OF_HOST_MEMORY:                 return "CL_OUT_OF_HOST_MEMORY";
	case CL_PROFILING_INFO_NOT_AVAILABLE:       return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case CL_MEM_COPY_OVERLAP:                   return "CL_MEM_COPY_OVERLAP";
	case CL_IMAGE_FORMAT_MISMATCH:              return "CL_IMAGE_FORMAT_MISMATCH";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case CL_BUILD_PROGRAM_FAILURE:              return "CL_BUILD_PROGRAM_FAILURE";
	case CL_MAP_FAILURE:                        return "CL_MAP_FAILURE";
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:       return "CL_MISALIGNED_SUB_BUFFER_OFFSET";                          //-13
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";   //-14
	case CL_COMPILE_PROGRAM_FAILURE:            return "CL_COMPILE_PROGRAM_FAILURE";                               //-15
	case CL_LINKER_NOT_AVAILABLE:               return "CL_LINKER_NOT_AVAILABLE";                                  //-16
	case CL_LINK_PROGRAM_FAILURE:               return "CL_LINK_PROGRAM_FAILURE";                                  //-17
	case CL_DEVICE_PARTITION_FAILED:            return "CL_DEVICE_PARTITION_FAILED";                               //-18
	case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";                         //-19
	case CL_INVALID_VALUE:                      return "CL_INVALID_VALUE";
	case CL_INVALID_DEVICE_TYPE:                return "CL_INVALID_DEVICE_TYPE";
	case CL_INVALID_PLATFORM:                   return "CL_INVALID_PLATFORM";
	case CL_INVALID_DEVICE:                     return "CL_INVALID_DEVICE";
	case CL_INVALID_CONTEXT:                    return "CL_INVALID_CONTEXT";
	case CL_INVALID_QUEUE_PROPERTIES:           return "CL_INVALID_QUEUE_PROPERTIES";
	case CL_INVALID_COMMAND_QUEUE:              return "CL_INVALID_COMMAND_QUEUE";
	case CL_INVALID_HOST_PTR:                   return "CL_INVALID_HOST_PTR";
	case CL_INVALID_MEM_OBJECT:                 return "CL_INVALID_MEM_OBJECT";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case CL_INVALID_IMAGE_SIZE:                 return "CL_INVALID_IMAGE_SIZE";
	case CL_INVALID_SAMPLER:                    return "CL_INVALID_SAMPLER";
	case CL_INVALID_BINARY:                     return "CL_INVALID_BINARY";
	case CL_INVALID_BUILD_OPTIONS:              return "CL_INVALID_BUILD_OPTIONS";
	case CL_INVALID_PROGRAM:                    return "CL_INVALID_PROGRAM";
	case CL_INVALID_PROGRAM_EXECUTABLE:         return "CL_INVALID_PROGRAM_EXECUTABLE";
	case CL_INVALID_KERNEL_NAME:                return "CL_INVALID_KERNEL_NAME";
	case CL_INVALID_KERNEL_DEFINITION:          return "CL_INVALID_KERNEL_DEFINITION";
	case CL_INVALID_KERNEL:                     return "CL_INVALID_KERNEL";
	case CL_INVALID_ARG_INDEX:                  return "CL_INVALID_ARG_INDEX";
	case CL_INVALID_ARG_VALUE:                  return "CL_INVALID_ARG_VALUE";
	case CL_INVALID_ARG_SIZE:                   return "CL_INVALID_ARG_SIZE";
	case CL_INVALID_KERNEL_ARGS:                return "CL_INVALID_KERNEL_ARGS";
	case CL_INVALID_WORK_DIMENSION:             return "CL_INVALID_WORK_DIMENSION";
	case CL_INVALID_WORK_GROUP_SIZE:            return "CL_INVALID_WORK_GROUP_SIZE";
	case CL_INVALID_WORK_ITEM_SIZE:             return "CL_INVALID_WORK_ITEM_SIZE";
	case CL_INVALID_GLOBAL_OFFSET:              return "CL_INVALID_GLOBAL_OFFSET";
	case CL_INVALID_EVENT_WAIT_LIST:            return "CL_INVALID_EVENT_WAIT_LIST";
	case CL_INVALID_EVENT:                      return "CL_INVALID_EVENT";
	case CL_INVALID_OPERATION:                  return "CL_INVALID_OPERATION";
	case CL_INVALID_GL_OBJECT:                  return "CL_INVALID_GL_OBJECT";
	case CL_INVALID_BUFFER_SIZE:                return "CL_INVALID_BUFFER_SIZE";
	case CL_INVALID_MIP_LEVEL:                  return "CL_INVALID_MIP_LEVEL";
	case CL_INVALID_GLOBAL_WORK_SIZE:           return "CL_INVALID_GLOBAL_WORK_SIZE";                           //-63
	case CL_INVALID_PROPERTY:                   return "CL_INVALID_PROPERTY";                                   //-64
	case CL_INVALID_IMAGE_DESCRIPTOR:           return "CL_INVALID_IMAGE_DESCRIPTOR";                           //-65
	case CL_INVALID_COMPILER_OPTIONS:           return "CL_INVALID_COMPILER_OPTIONS";                           //-66
	case CL_INVALID_LINKER_OPTIONS:             return "CL_INVALID_LINKER_OPTIONS";                             //-67
	case CL_INVALID_DEVICE_PARTITION_COUNT:     return "CL_INVALID_DEVICE_PARTITION_COUNT";                     //-68
	default:
		return "UNKNOWN ERROR CODE";
	}
}


inline std::string getDeviceInfoString(cl_device_id dev_id, cl_device_info info)
{
	size_t size;
	clGetDeviceInfo(dev_id, info, 0, nullptr, &size);
	std::string ret(size, '\0');
	clGetDeviceInfo(dev_id, info, size, &ret[0], nullptr);
	return ret;
}

OpenCLMgr::OpenCLMgr()
{
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	for (auto& p : platforms) 
	{
		std::vector<cl::Device> devices;
		p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
		for (auto& d : devices) 
		{
			std::string deviceName = d.getInfo<CL_DEVICE_NAME>();
			std::string deviceVersion = d.getInfo<CL_DEVICE_VERSION>();
			std::string driverVersion = d.getInfo<CL_DRIVER_VERSION>();
			std::string deviceOpenCLVersion = d.getInfo<CL_DEVICE_OPENCL_C_VERSION>();
			cl_uint maxComputeUnits = d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
			size_t workitem_dims = d.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
			cl::vector<size_t> workitem_size = d.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
			size_t workgroup_size = d.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
			cl_uint clock_freq = d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
			ndevs.emplace_back(DeviceInfo{ deviceName, deviceVersion, driverVersion, deviceOpenCLVersion, maxComputeUnits, d });
		}
	}
}

OpenCLMgr::~OpenCLMgr()
{
	releaseCL();
}

const OpenCLMgr::DeviceInfo* OpenCLMgr::createDevice(size_t id)
{
	cl_int err = 0;
	if (id >= ndevs.size()) return nullptr;
	clDevice = ndevs[id].clDevice;
	clContext = cl::Context({ clDevice });
	clQueue = cl::CommandQueue(clContext, clDevice);
	return &ndevs[id];
}

int OpenCLMgr::buildProgram(const std::vector<std::string>& sources, const std::string & option, std::string * err)
{
	clProgram = cl::Program(clContext, sources);
	try
	{
		clProgram.build({ clDevice }, option.c_str());
	}
	catch (...)
	{
		cl_int buildErr = CL_SUCCESS;
		auto buildInfo = clProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
		if (err)
		{
			*err = "";
			for (auto &pair : buildInfo)
			{
				*err += pair.second + "\n\n";
			}
		}
		if (!buildErr) buildErr = -1;
		return buildErr;
	}
	return 0;
}

int OpenCLMgr::buildProgramFromFile(const std::vector<std::string>& filePaths, const std::string& source, const std::string & option, std::string * err)
{
	std::vector<std::string> sources;
	if (!source.empty()) sources.emplace_back(source);
	for (auto && filePath : filePaths)
	{
		std::ifstream ifs{ filePath };
		if (!ifs)
		{
			*err += "cannot find file '" + filePath + "'\n";
			return -1;
		}
		std::string source, line;
		while (getline(ifs, line))
		{
			source += line + "\n";
		}
		sources.emplace_back(move(source));
	}
	return buildProgram(sources, option, err);
}

int OpenCLMgr::writeProgramBinary(const char * name) const
{
	cl::vector<cl::vector<unsigned char>> bins = clProgram.getInfo<CL_PROGRAM_BINARIES>();
	std::ofstream of{ name, std::ios_base::binary };
	of.write((const char*)&bins[0][0], bins[0].size());
	return 0;
}

void OpenCLMgr::releaseCL()
{
}
