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

		Structure::Structure(const Init& init)
		{
			auto context = GetContext();

			OptixAccelBuildOptions accel_options = {};
			accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
			accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

			OptixBuildInput input = init.input;
			std::unique_ptr<CUDA::DeviceMemory> deviceVertices;
			CUdeviceptr memory = {};
			switch (input.type)
			{
			case OPTIX_BUILD_INPUT_TYPE_TRIANGLES:
				deviceVertices = std::make_unique<CUDA::DeviceMemory>(init.data);
				memory = deviceVertices->GetCuDevPtr();
				input.triangleArray.vertexBuffers = &memory;
				input.triangleArray.flags = init.flags.data();
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
		pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
		pipelineOptions.numPayloadValues = numPayloadValues;
		pipelineOptions.numAttributeValues = numAttribValues;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
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


	void ModuleInit::AddProgram(OptixProgramGroupKind kind, std::string entryFunctionName)
	{
		programs.push_back({ .kind = kind, .entryFunctionName = std::move(entryFunctionName) });
	}


	Module::Module(const ModuleInit& init)
	{
		pipelineOptions = init.pipelineOptions;

		OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
			GetContext(),
			&init.moduleOptions,
			&init.pipelineOptions,
			reinterpret_cast<const char*>(init.ptxData.data()),
			init.ptxData.size(),
			LOG, &LOG_SIZE,
			&module
		));

		programsBlock.reserve(init.programs.size());
		programs.reserve(init.programs.size());

		for (uint32_t i = 0; i < init.programs.size(); i++)
		{
			auto& programInfo = init.programs[i];
			auto& program = programs.emplace_back();

			OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

			OptixProgramGroupDesc program_desc = {};
			program_desc.kind = programInfo.kind;
			program.kind = programInfo.kind;

			switch (programInfo.kind)
			{
			case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
				program_desc.raygen.module = module;
				program_desc.raygen.entryFunctionName = programInfo.entryFunctionName.c_str();
				break;
			case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
				program_desc.hitgroup.moduleCH = module;
				program_desc.hitgroup.entryFunctionNameCH = programInfo.entryFunctionName.c_str();
				break;
			case OPTIX_PROGRAM_GROUP_KIND_MISS:
				program_desc.miss.module = module;
				program_desc.miss.entryFunctionName = programInfo.entryFunctionName.c_str();
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

	OptixProgramGroup Module::GetProgram(OptixProgramGroupKind kind) const
	{
		auto it = std::find_if(programs.begin(), programs.end(), [kind](auto& p) { return p.kind == kind; });

		if (it == programs.end())
		{
			throw Exception("Can't find program group kind in the module");
		}

		return it->program;
	}



	PipelineInit::PipelineInit(const Module& module, uint32_t maxTraceDepth)
		: module(module)
		, maxTraceDepth(maxTraceDepth)
	{

	}

	void PipelineInit::AddSbtValue(std::span<const uint8_t> data, OptixProgramGroupKind kind, size_t stride)
	{
		if (stride == 0)
		{
			stride = data.size_bytes();
		}

		auto it = std::find_if(sbtValues.begin(), sbtValues.end(), [kind](auto& sbt) { return sbt.kind == kind; });
		if (it != sbtValues.end())
		{
			throw Exception("SbtValue kind already exists");
		}

		sbtValues.push_back(SbtValue{ .data = { data.begin(), data.end() }, .kind = kind, .stride = stride });
	}

	Pipeline::Pipeline(const PipelineInit& init)
		: module(init.module)
	{
		OptixPipelineLinkOptions pipeline_link_options = {};
		pipeline_link_options.maxTraceDepth = init.maxTraceDepth;
		pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
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
			OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
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
		for (auto& sbtValue : init.sbtValues)
		{
			const size_t recordCount = sbtValue.data.size() / sbtValue.stride;

			localSbtValue = sbtValue.data;
			for (uint32_t i = 0; i < recordCount; i++)
			{
				OPTIX_CHECK(optixSbtRecordPackHeader(module.GetProgram(sbtValue.kind), localSbtValue.data() + i * sbtValue.stride));
			}

			auto memory = std::make_unique<CUDA::DeviceMemory>(localSbtValue);

			switch (sbtValue.kind)
			{
			case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
				shaderBindingTable.raygenRecord = memory->GetCuDevPtr();
				break;
			case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
				shaderBindingTable.hitgroupRecordBase = memory->GetCuDevPtr();
				shaderBindingTable.hitgroupRecordStrideInBytes = (uint32_t)sbtValue.stride;
				shaderBindingTable.hitgroupRecordCount = (uint32_t)recordCount;
				break;
			case OPTIX_PROGRAM_GROUP_KIND_MISS:
				shaderBindingTable.missRecordBase = memory->GetCuDevPtr();
				shaderBindingTable.missRecordStrideInBytes = (uint32_t)sbtValue.stride;
				shaderBindingTable.missRecordCount = (uint32_t)recordCount;
				break;
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