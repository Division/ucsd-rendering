#include "OptixHelper.h"
#include <optix_function_table_definition.h>
#include "FileLoader.h"

namespace Optix
{
	namespace
	{
		OptixDeviceContext _context = nullptr;
		CUstream _stream;

		void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
		{
			std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
				<< message << "\n";
		}
	}


	OptixDeviceContext GetContext()
	{
		return _context;
	}


	CUstream GetStream()
	{
		return _stream;
	}


	namespace Acceleration
	{
		Init Init::Triangles(std::span<const glm::vec3> vertices)
		{
			Init result;
			result.data = std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(vertices.data()), vertices.size_bytes());
			result.input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
			result.input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			result.input.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
			result.input.triangleArray.numSbtRecords = 1;
			result.flags[0] = OPTIX_GEOMETRY_FLAG_NONE;

			return result;
		}

		Init Init::Spheres(std::span<const Device::Scene::Sphere> spheres)
		{
			Init result;
			result.data = std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(spheres.data()), spheres.size_bytes());
			result.input = {};
			result.input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
			result.input.sphereArray.numVertices = static_cast<uint32_t>(spheres.size());
			result.input.sphereArray.numSbtRecords = 1;
			result.input.sphereArray.primitiveIndexOffset = 0;
			result.input.sphereArray.singleRadius = 0;
			result.input.sphereArray.sbtIndexOffsetBuffer = 0;
			result.input.sphereArray.radiusStrideInBytes = sizeof(Device::Scene::Sphere);
			result.input.sphereArray.vertexStrideInBytes = sizeof(Device::Scene::Sphere);
			result.flags[0] = OPTIX_GEOMETRY_FLAG_NONE;

			return result;
		}

		Init Init::Instances(std::span<const Instance> inputInstances)
		{
			Init result;
			result.instances.resize(inputInstances.size());
			for (uint32_t i = 0; i < inputInstances.size(); i++)
			{
				auto& inst = result.instances[i];
				inst = {};
				inst.visibilityMask = 255;
				inst.sbtOffset = inputInstances[i].sbtOffset;
				auto rowMajor = glm::transpose(inputInstances[i].transform);
				memcpy(inst.transform, &rowMajor, sizeof(inst.transform));
				inst.instanceId = i;
				inst.traversableHandle = inputInstances[i].structure->GetHandle();
			}

			result.data = std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(result.instances.data()), result.instances.size() * sizeof(OptixInstance));
			result.input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
			result.input.instanceArray.instanceStride = 0;
			result.input.instanceArray.numInstances = static_cast<uint32_t>(inputInstances.size());

			return result;
		}

		Structure::Structure(const Init& init)
		{
			auto context = GetContext();

			OptixAccelBuildOptions accel_options = {};
			accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
			accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

			OptixBuildInput input = init.input;
			std::unique_ptr<CUDA::DeviceMemory> deviceDataBuffer;
			std::unique_ptr<CUDA::DeviceMemory> radiusDataBuffer;
			CUdeviceptr memory = {};
			CUdeviceptr radiusMemory = {};
			switch (input.type)
			{
			case OPTIX_BUILD_INPUT_TYPE_TRIANGLES:
				deviceDataBuffer = std::make_unique<CUDA::DeviceMemory>(init.data);
				memory = deviceDataBuffer->GetCuDevPtr();
				input.triangleArray.vertexBuffers = &memory;
				input.triangleArray.flags = init.flags.data();
				break;
				
			case OPTIX_BUILD_INPUT_TYPE_SPHERES:
				deviceDataBuffer = std::make_unique<CUDA::DeviceMemory>(init.data);
				memory = deviceDataBuffer->GetCuDevPtr();
				radiusMemory = deviceDataBuffer->GetCuDevPtr() + sizeof(glm::vec3);
				input.sphereArray.vertexBuffers = &memory;
				input.sphereArray.radiusBuffers = &radiusMemory;
				input.sphereArray.flags = init.flags.data();
				break;

			case OPTIX_BUILD_INPUT_TYPE_INSTANCES:
				deviceDataBuffer = std::make_unique<CUDA::DeviceMemory>(init.data);
				input.instanceArray.instances = deviceDataBuffer->GetCuDevPtr();
				break;

			default:
				throw Exception("Unknown build input type");
			}

			OptixAccelBufferSizes bufferSizes;
			OPTIX_CHECK(optixAccelComputeMemoryUsage(
				context,
				&accel_options,
				&input,
				1, // Number of build inputs
				&bufferSizes
			));

			CUDA::DeviceMemory tempBuffer(bufferSizes.tempSizeInBytes);
			gpuBuffer = std::make_unique<CUDA::DeviceMemory>(bufferSizes.outputSizeInBytes);

			OPTIX_CHECK(optixAccelBuild(
				context,
				0,                  // CUDA stream
				&accel_options,
				&input,
				1,                  // num build inputs
				tempBuffer.GetCuDevPtr(),
				bufferSizes.tempSizeInBytes,
				gpuBuffer->GetCuDevPtr(),
				bufferSizes.outputSizeInBytes,
				&accHandle,
				nullptr,            // emitted property list
				0                   // num emitted properties
			));

		}

	}


	ModuleInit::ModuleInit(const std::wstring& ptxPath, uint32_t numPayloadValues, uint32_t numAttribValues, uint32_t primitiveTypeFlags)
	{
#if !defined( NDEBUG )
		moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
		moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

		pipelineOptions.usesMotionBlur = false;
		pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;// OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS | OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
		pipelineOptions.numPayloadValues = numPayloadValues;
		pipelineOptions.numAttributeValues = numAttribValues;
#ifdef _DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
		pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
		pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
		pipelineOptions.pipelineLaunchParamsVariableName = "params";
		pipelineOptions.usesPrimitiveTypeFlags = primitiveTypeFlags;

		auto result = Loader::LoadFile(ptxPath);
		if (!result)
		{
			throw Exception("Failed loading ptx file");
		}

		ptxData = std::move(*result);
	}

	uint32_t ModuleInit::AddRaygenProgram(std::string entryFunctionName)
	{
		programs.push_back(std::make_unique<Program>());
		auto& program = *programs.back();
		program.entryFunctionName0 = std::move(entryFunctionName);
		program.desc = {};
		program.desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		program.desc.raygen.entryFunctionName = program.entryFunctionName0.c_str();
		return static_cast<uint32_t>(programs.size() - 1);
	}

	uint32_t ModuleInit::AddMissProgram(std::string entryFunctionName)
	{
		programs.push_back(std::make_unique<Program>());
		auto& program = *programs.back();
		program.entryFunctionName0 = std::move(entryFunctionName);
		program.desc = {};
		program.desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		program.desc.miss.entryFunctionName = program.entryFunctionName0.c_str();
		return static_cast<uint32_t>(programs.size() - 1);
	}

	uint32_t ModuleInit::AddHitProgram(std::string entryFunctionCH, std::string entryFunctionAH, std::string entryFunctionIS, bool isSpheres)
	{
		programs.push_back(std::make_unique<Program>());
		auto& program = *programs.back();
		program.entryFunctionName0 = entryFunctionCH;
		program.entryFunctionName1 = entryFunctionAH;
		program.entryFunctionName2 = entryFunctionIS;
		program.desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

		if (program.entryFunctionName0.size())
			program.desc.hitgroup.entryFunctionNameCH = program.entryFunctionName0.c_str();
		if (program.entryFunctionName1.size())
			program.desc.hitgroup.entryFunctionNameAH = program.entryFunctionName1.c_str();
		if (program.entryFunctionName2.size())
			program.desc.hitgroup.entryFunctionNameIS = program.entryFunctionName2.c_str();

		program.isSpheres = isSpheres;

		return static_cast<uint32_t>(programs.size() - 1);
	}


	Module::Module(const ModuleInit& init)
	{
		pipelineOptions = init.pipelineOptions;

		
		OPTIX_CHECK_LOG(optixModuleCreate(
			GetContext(),
			&init.moduleOptions,
			&init.pipelineOptions,
			reinterpret_cast<const char*>(init.ptxData.data()),
			init.ptxData.size(),
			LOG, &LOG_SIZE,
			&module
		));

        OptixModule sphereModule = nullptr;
		OptixBuiltinISOptions builtin_is_options = {};

		builtin_is_options.usesMotionBlur = false;
		builtin_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
		OPTIX_CHECK_LOG(optixBuiltinISModuleGet(GetContext(), &init.moduleOptions, &init.pipelineOptions,
			&builtin_is_options, &sphereModule));

		programsBlock.reserve(init.programs.size());
		programs.reserve(init.programs.size());

		for (uint32_t i = 0; i < init.programs.size(); i++)
		{
			const auto& programInfo = init.programs[i];
			auto& program = programs.emplace_back();

			OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

			OptixProgramGroupDesc program_desc = programInfo->desc;
			program.kind = program_desc.kind;

			switch (program_desc.kind)
			{
			case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
				program_desc.raygen.module = module;
				break;
			case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
				program_desc.hitgroup.moduleCH = module;

				if (programInfo->isSpheres)
				{
					program_desc.hitgroup.entryFunctionNameIS = nullptr;
					program_desc.hitgroup.moduleIS = sphereModule;
				}
				break;
			case OPTIX_PROGRAM_GROUP_KIND_MISS:
				program_desc.miss.module = module;
				break;
			}

			OPTIX_CHECK_LOG(optixProgramGroupCreate(
				GetContext(),
				&program_desc,
				1,   // num program groups
				&program_group_options,
				LOG, &LOG_SIZE,
				&program.program
			));

			programsBlock.push_back(program.program);
		}
	}

	const Module::Program& Module::GetProgram(uint32_t index) const
	{
		if (index >= programs.size())
		{
			throw Exception("Can't find program index in the module");
		}

		return programs[index];
	}



	PipelineInit::PipelineInit(const Module& module, uint32_t maxTraceDepth)
		: module(module)
		, maxTraceDepth(maxTraceDepth)
	{
	}

	void PipelineInit::AddSbtValue(OptixProgramGroupKind kind, uint32_t programIndex, std::span<const uint8_t> data, size_t stride)
	{
		if (stride == 0)
		{
			stride = data.size_bytes();
		}

		if (data.size_bytes() % stride != 0)
			throw Exception("Wrong stride or data size");

		auto it = sbtValues.find(kind);
		if (it == sbtValues.end())
			it = sbtValues.insert({ kind, SbtValue{} }).first;
		else if (it->second.stride != stride)
			throw Exception("Stride mismatch");

		SbtValue& sbtValue = it->second;
		sbtValue.stride = stride;
		sbtValue.kind = kind;
		sbtValue.data.reserve(sbtValue.data.size() + data.size());
		sbtValue.data.insert(sbtValue.data.end(), data.begin(), data.end());
		const uint32_t count = static_cast<uint32_t>(data.size_bytes() / stride);
		for (uint32_t i = 0; i < count; i++)
			sbtValue.indices.push_back(programIndex);
	}

	Pipeline::Pipeline(const PipelineInit& init)
		: module(init.module)
	{
		OptixPipelineLinkOptions pipeline_link_options = {};
		pipeline_link_options.maxTraceDepth = init.maxTraceDepth;
		OPTIX_CHECK_LOG(optixPipelineCreate(
			GetContext(),
			&module.GetPipelineOptions(),
			&pipeline_link_options,
			module.GetPrograms().data(),
			(uint32_t)module.GetPrograms().size(),
			LOG, &LOG_SIZE,
			&pipeline
		));

		OptixStackSizes stack_sizes = {};
		for (auto& prog_group : module.GetPrograms())
		{
			OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, nullptr));
		}

		uint32_t direct_callable_stack_size_from_traversal;
		uint32_t direct_callable_stack_size_from_state;
		uint32_t continuation_stack_size;
		OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, init.maxTraceDepth,
			0,  // maxCCDepth
			0,  // maxDCDEpth
			&direct_callable_stack_size_from_traversal,
			&direct_callable_stack_size_from_state, &continuation_stack_size));
		OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
			direct_callable_stack_size_from_state, continuation_stack_size,
			1  // maxTraversableDepth
		));

		// Setting up shader binding table


		std::vector<uint8_t> localSbtValue;
		for (auto& it : init.sbtValues)
		{

			auto& sbtValue = it.second;

			const uint32_t recordStride = static_cast<uint32_t>(sbtValue.stride);
			const uint32_t recordCount = static_cast<uint32_t>(sbtValue.data.size() / recordStride);
			
			if (!sbtValue.data.empty())
			{
				localSbtValue = sbtValue.data;

				for (uint32_t i = 0; i < recordCount; i++)
				{
					OPTIX_CHECK(optixSbtRecordPackHeader(module.GetProgram(sbtValue.indices[i]).program, localSbtValue.data() + i * recordStride));
				}
			}
			else
			{
				throw Exception("Sbt data can't be empty");
			}

			auto memory = std::make_unique<CUDA::DeviceMemory>(localSbtValue);

			switch (sbtValue.kind)
			{
			case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
				shaderBindingTable.raygenRecord = memory->GetCuDevPtr();
				break;
			case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
				shaderBindingTable.hitgroupRecordBase = memory->GetCuDevPtr();
				shaderBindingTable.hitgroupRecordStrideInBytes = (uint32_t)recordStride;
				shaderBindingTable.hitgroupRecordCount = (uint32_t)recordCount;
				break;
			case OPTIX_PROGRAM_GROUP_KIND_MISS:
				shaderBindingTable.missRecordBase = memory->GetCuDevPtr();
				shaderBindingTable.missRecordStrideInBytes = (uint32_t)recordStride;
				shaderBindingTable.missRecordCount = (uint32_t)recordCount;
				break;
			default:
				throw Exception("Unknown sbt program kind");
			}
			sbtMemory.push_back(SbtData{ .memory = std::move(memory) });
		}
	}



	void Launch(std::span<const uint8_t> params, const Pipeline& pipeline, uint32_t width, const uint32_t height, const uint32_t depth)
	{
		CUDA::DeviceMemory paramMemory(params);

		OPTIX_CHECK(optixLaunch(pipeline.GetPipeline(), GetStream(), paramMemory.GetCuDevPtr(), params.size_bytes(), &pipeline.GetSBT(), width, height, depth));
		CUDA_SYNC_CHECK();
	}




	bool Initialize()
	{
		// Initialize CUDA
		CUDA_CHECK(cudaFree(0));

		CUcontext cuCtx = 0;  // zero means take the current context
		OPTIX_CHECK(optixInit());
		OptixDeviceContextOptions options = {};
		options.logCallbackFunction = &context_log_cb;
		options.logCallbackLevel = 4;
		OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &_context));

		CUDA_CHECK(cudaStreamCreate(&_stream));

		return true;

	}


	void Deinitialize()
	{

	}

}