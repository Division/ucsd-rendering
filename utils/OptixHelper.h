#pragma once

#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include "CUDAHelper.h"

namespace Optix
{

	template <typename T>
	struct SbtRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		T data;
	};



	class Exception : public std::runtime_error
	{
	public:
		Exception(const char* message) : std::runtime_error(message) {}
		Exception(const std::string& message) : std::runtime_error(message) {}
	};


	bool Initialize();
	void Deinitialize();
	OptixDeviceContext GetContext();

	namespace Acceleration
	{

		struct Init
		{
			friend class Structure;

			struct Instance
			{
				glm::mat4 transform;
				Structure* structure;
			};

		private:
			std::vector<OptixInstance> instances;
			std::span<const uint8_t> data;
			OptixBuildInput input = {};
			std::array<uint32_t, 8> flags;

		public:
			static Init Triangles(std::span<const glm::vec3> vertices);
			static Init Instances(std::span<const Instance> inputInstances);
		};

		class Structure
		{
			std::unique_ptr<CUDA::DeviceMemory> gpuBuffer;
			OptixTraversableHandle accHandle;

		public:
			Structure(const Init& init);

			OptixTraversableHandle GetHandle() const { return accHandle; }
			CUDA::DeviceMemory& GetBuffer() const { return *gpuBuffer; }

		};

	}



	struct ModuleInit
	{
		struct Program
		{
			OptixProgramGroupKind kind;
			std::string entryFunctionName;
		};

		OptixPipelineCompileOptions pipelineOptions = {};
		OptixModuleCompileOptions moduleOptions = {};
		std::vector<uint8_t> ptxData;
		std::vector<Program> programs;

		void AddProgram(OptixProgramGroupKind kind, std::string entryFunctionName);

		ModuleInit(const std::wstring& ptxPath, uint32_t numPayloadValues, uint32_t numAttribValues, uint32_t primitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);
	};

	class Module
	{
		OptixModule module;

		struct Program
		{
			OptixProgramGroup program;
			OptixProgramGroupKind kind;
		};

		std::vector<Program> programs;
		std::vector<OptixProgramGroup> programsBlock;
		OptixPipelineCompileOptions pipelineOptions;

	public:
		Module(const ModuleInit& init);

		OptixProgramGroup GetProgram(OptixProgramGroupKind kind) const;
		std::span<const OptixProgramGroup> GetPrograms() const { return programsBlock; }
		OptixModule GetModule() const { return module; }
		const OptixPipelineCompileOptions& GetPipelineOptions() const { return pipelineOptions; }
	};



	struct PipelineInit
	{
		const Module& module;
		uint32_t maxTraceDepth;
		struct SbtValue
		{
			std::vector<uint8_t> data;
			OptixProgramGroupKind kind;
			size_t stride;
		};

		std::vector<SbtValue> sbtValues;

		PipelineInit(const Module& module, uint32_t maxTraceDepth);

		void AddSbtValue(std::span<const uint8_t> data, OptixProgramGroupKind kind, size_t stride = 0);

		template<typename T>
		void AddSbtValue(const SbtRecord<T>& value, OptixProgramGroupKind kind)
		{
			AddSbtValue(std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&value), sizeof(value)), kind);
		}

		template<typename T>
		void AddSbtValues(const std::span<const SbtRecord<T>> value, OptixProgramGroupKind kind)
		{
			AddSbtValue(std::span<const uint8_t>(value.data(), value.size_bytes()), kind, sizeof(SbtRecord<T>));
		}
	};

	class Pipeline
	{
		const Module& module;
		uint32_t maxTraceDepth;
		OptixPipeline pipeline = nullptr;
		OptixShaderBindingTable shaderBindingTable = {};

		struct SbtData
		{
			std::unique_ptr<CUDA::DeviceMemory> memory;
		};

		std::vector<SbtData> sbtMemory;

	public:
		Pipeline(const PipelineInit& init);

		OptixPipeline GetPipeline() const { return pipeline; }
		const OptixShaderBindingTable& GetSBT() const { return shaderBindingTable; }
	};


	void Launch(std::span<const uint8_t> params, const Pipeline& pipeline, uint32_t width, const uint32_t height, const uint32_t depth = 1);

	template<typename T>
	void Launch(const T& params, const Pipeline& pipeline, uint32_t width, const uint32_t height, const uint32_t depth = 1)
	{
		Launch({ reinterpret_cast<const uint8_t*>(&params), sizeof(params) }, pipeline, width, height, depth);
	}
}
