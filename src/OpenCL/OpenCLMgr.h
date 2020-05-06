#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include "opencl_dyn_load.h"
#include <CL/cl2.hpp>
#include <vector>
#include <string>

namespace tomoto
{
	namespace ocl
	{
		class OpenCLMgr
		{
		public:
			struct DeviceInfo
			{
				std::string name, version, driver, openCL;
				cl_uint maxComputeUnits;
				cl::Device clDevice;
			};
		protected:
			std::vector<DeviceInfo> ndevs;
			cl::Device clDevice;
			cl::Context clContext;
			cl::CommandQueue clQueue;
			cl::Program clProgram;
		public:
			OpenCLMgr();
			~OpenCLMgr();
			const DeviceInfo* createDevice(size_t id);
			const std::vector<DeviceInfo>& getDeviceList() const { return ndevs; }
			int buildProgram(const std::vector<std::string>& sources, const std::string& option = "", std::string* err = nullptr);
			int buildProgramFromFile(const std::vector<std::string>& filePath, const std::string& source = "", const std::string& option = "", std::string* err = nullptr);
			template<class ...Type>
			cl::KernelFunctor<Type...> getKernelFunc(const std::string& name) const
			{
				return cl::KernelFunctor<Type...>(clProgram, name);
			}
			int writeProgramBinary(const char* name) const;
			void releaseCL();
			cl::Device& getDevice() { return clDevice; }
			cl::Context& getContext() { return clContext; }
			cl::CommandQueue& getQueue() { return clQueue; }
			const cl::CommandQueue& getQueue() const { return clQueue; }
			static const char* translateOpenCLError(cl_int errorCode);
		};
	}
}