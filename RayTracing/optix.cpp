#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <sutil/Trackball.h>
#include "utils/FileLoader.h"
#include "utils/CUDAHelper.h"
#include <sutil/vec_math.h>


namespace
{
	class Camera {
public:
    Camera()
        : m_eye(make_float3(1.0f)), m_lookat(make_float3(0.0f)), m_up(make_float3(0.0f, 1.0f, 0.0f)), m_fovY(35.0f), m_aspectRatio(1.0f)
    {
    }

    Camera(const float3& eye, const float3& lookat, const float3& up, float fovY, float aspectRatio)
        : m_eye(eye), m_lookat(lookat), m_up(up), m_fovY(fovY), m_aspectRatio(aspectRatio)
    {
    }

    float3 direction() const { return normalize(m_lookat - m_eye); }
    void setDirection(const float3& dir) { m_lookat = m_eye + length(m_lookat - m_eye) * dir; }

    const float3& eye() const { return m_eye; }
    void setEye(const float3& val) { m_eye = val; }
    const float3& lookat() const { return m_lookat; }
    void setLookat(const float3& val) { m_lookat = val; }
    const float3& up() const { return m_up; }
    void setUp(const float3& val) { m_up = val; }
    const float& fovY() const { return m_fovY; }
    void setFovY(const float& val) { m_fovY = val; }
    const float& aspectRatio() const { return m_aspectRatio; }
    void setAspectRatio(const float& val) { m_aspectRatio = val; }

    // UVW forms an orthogonal, but not orthonormal basis!
	void UVWFrame(float3& U, float3& V, float3& W) const
	{
		W = m_lookat - m_eye; // Do not normalize W -- it implies focal length
		float wlen = length(W);
		U = normalize(cross(W, m_up));
		V = normalize(cross(U, W));

		float vlen = wlen * tanf(0.5f * m_fovY * M_PIf / 180.0f);
		V *= vlen;
		float ulen = vlen * m_aspectRatio;
		U *= ulen;
	}

private:
    float3 m_eye;
    float3 m_lookat;
    float3 m_up;
    float m_fovY;
    float m_aspectRatio;
};

}

namespace CUDA
{
	static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
	{
		std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
			<< message << "\n";
	}

	struct Params
	{
		void* image;
		unsigned int           image_width;
		unsigned int           image_height;
		unsigned int           pitch;
		float3                 cam_eye;
		float3                 cam_u, cam_v, cam_w;
		OptixTraversableHandle handle;
	};


	struct RayGenData
	{
		// No data needed
	};


	struct MissData
	{
		float3 bg_color;
	};


	struct HitGroupData
	{
		// No data needed
	};

	template <typename T>
	struct SbtRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		T data;
	};

	typedef SbtRecord<RayGenData>     RayGenSbtRecord;
	typedef SbtRecord<MissData>       MissSbtRecord;
	typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;

	void configureCamera(Camera& cam, const uint32_t width, const uint32_t height)
	{
		cam.setEye({ 0.0f, 0.0f, 2.0f });
		cam.setLookat({ 0.0f, 0.0f, 0.0f });
		cam.setUp({ 0.0f, 1.0f, 3.0f });
		cam.setFovY(45.0f);
		cam.setAspectRatio((float)width / (float)height);
	}


	void SetupOptix(const uint32_t width, const uint32_t height, const uint32_t pitch, void* outputBuffer)
	{
		OptixDeviceContext context = nullptr;
		{
			// Initialize CUDA
			CUDA_CHECK(cudaFree(0));

			CUcontext cuCtx = 0;  // zero means take the current context
			OPTIX_CHECK(optixInit());
			OptixDeviceContextOptions options = {};
			options.logCallbackFunction = &context_log_cb;
			options.logCallbackLevel = 4;
			OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
		}




		//
		// accel handling
		//
		OptixTraversableHandle gas_handle;
		CUdeviceptr            d_gas_output_buffer;
		{
			// Use default options for simplicity.  In a real use case we would want to
			// enable compaction, etc
			OptixAccelBuildOptions accel_options = {};
			accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
			accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

			// Triangle build input: simple list of three vertices
			const std::array<float3, 3> vertices =
			{ {
				  { -0.5f, -0.5f, 0.0f },
				  {  0.5f, -0.5f, 0.0f },
				  {  0.0f,  0.5f, 0.0f }
			} };

			const size_t vertices_size = sizeof(float3) * vertices.size();
			CUdeviceptr d_vertices = 0;
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices_size));
			CUDA_CHECK(cudaMemcpy(
				reinterpret_cast<void*>(d_vertices),
				vertices.data(),
				vertices_size,
				cudaMemcpyHostToDevice
			));

			// Our build input is a simple list of non-indexed triangle vertices
			const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
			OptixBuildInput triangle_input = {};
			triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
			triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			triangle_input.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
			triangle_input.triangleArray.vertexBuffers = &d_vertices;
			triangle_input.triangleArray.flags = triangle_input_flags;
			triangle_input.triangleArray.numSbtRecords = 1;

			OptixAccelBufferSizes gas_buffer_sizes;
			OPTIX_CHECK(optixAccelComputeMemoryUsage(
				context,
				&accel_options,
				&triangle_input,
				1, // Number of build inputs
				&gas_buffer_sizes
			));
			CUdeviceptr d_temp_buffer_gas;
			CUDA_CHECK(cudaMalloc(
				reinterpret_cast<void**>(&d_temp_buffer_gas),
				gas_buffer_sizes.tempSizeInBytes
			));
			CUDA_CHECK(cudaMalloc(
				reinterpret_cast<void**>(&d_gas_output_buffer),
				gas_buffer_sizes.outputSizeInBytes
			));

			OPTIX_CHECK(optixAccelBuild(
				context,
				0,                  // CUDA stream
				&accel_options,
				&triangle_input,
				1,                  // num build inputs
				d_temp_buffer_gas,
				gas_buffer_sizes.tempSizeInBytes,
				d_gas_output_buffer,
				gas_buffer_sizes.outputSizeInBytes,
				&gas_handle,
				nullptr,            // emitted property list
				0                   // num emitted properties
			));

			// We can now free the scratch space buffer used during build and the vertex
			// inputs, since they are not needed by our trivial shading method
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));
		}



		// "D:\code\ucsd-rendering\projects\01-raytracing\x64\Debug\triangle.cu.ptx"	

			//
			// Create module
			//
		OptixModule module = nullptr;
		OptixPipelineCompileOptions pipeline_compile_options = {};
		{
			OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
			module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
			module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

			pipeline_compile_options.usesMotionBlur = false;
			pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
			pipeline_compile_options.numPayloadValues = 3;
			pipeline_compile_options.numAttributeValues = 3;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
			pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
			pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
			pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
			pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;


			auto data = Loader::LoadFile(L"D:/code/ucsd-rendering/projects/01-raytracing/x64/Debug/triangle.cu.obj");
			//auto data = Loader::LoadFile(L"D:/code/optix_samples/lib/ptx/Debug/optixTriangle_generated_optixTriangle.cu.optixir");
			const size_t      inputSize = data->size();


			const char* input = (const char*)data->data();

			OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
				context,
				&module_compile_options,
				&pipeline_compile_options,
				input,
				inputSize,
				LOG, &LOG_SIZE,
				&module
			));



			//
			// Create program groups
			//
			OptixProgramGroup raygen_prog_group = nullptr;
			OptixProgramGroup miss_prog_group = nullptr;
			OptixProgramGroup hitgroup_prog_group = nullptr;
			{
				OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

				OptixProgramGroupDesc raygen_prog_group_desc = {}; //
				raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
				raygen_prog_group_desc.raygen.module = module;
				raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
				OPTIX_CHECK_LOG(optixProgramGroupCreate(
					context,
					&raygen_prog_group_desc,
					1,   // num program groups
					&program_group_options,
					LOG, &LOG_SIZE,
					&raygen_prog_group
				));

				OptixProgramGroupDesc miss_prog_group_desc = {};
				miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
				miss_prog_group_desc.miss.module = module;
				miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
				OPTIX_CHECK_LOG(optixProgramGroupCreate(
					context,
					&miss_prog_group_desc,
					1,   // num program groups
					&program_group_options,
					LOG, &LOG_SIZE,
					&miss_prog_group
				));

				OptixProgramGroupDesc hitgroup_prog_group_desc = {};
				hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
				hitgroup_prog_group_desc.hitgroup.moduleCH = module;
				hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
				OPTIX_CHECK_LOG(optixProgramGroupCreate(
					context,
					&hitgroup_prog_group_desc,
					1,   // num program groups
					&program_group_options,
					LOG, &LOG_SIZE,
					&hitgroup_prog_group
				));
			}

			//
			// Link pipeline
			//
			OptixPipeline pipeline = nullptr;
			{
				const uint32_t    max_trace_depth = 1;
				OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

				OptixPipelineLinkOptions pipeline_link_options = {};
				pipeline_link_options.maxTraceDepth = max_trace_depth;
				pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
				OPTIX_CHECK_LOG(optixPipelineCreate(
					context,
					&pipeline_compile_options,
					&pipeline_link_options,
					program_groups,
					sizeof(program_groups) / sizeof(program_groups[0]),
					LOG, &LOG_SIZE,
					&pipeline
				));

				OptixStackSizes stack_sizes = {};
				for (auto& prog_group : program_groups)
				{
					OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
				}

				uint32_t direct_callable_stack_size_from_traversal;
				uint32_t direct_callable_stack_size_from_state;
				uint32_t continuation_stack_size;
				OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
					0,  // maxCCDepth
					0,  // maxDCDEpth
					&direct_callable_stack_size_from_traversal,
					&direct_callable_stack_size_from_state, &continuation_stack_size));
				OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
					direct_callable_stack_size_from_state, continuation_stack_size,
					1  // maxTraversableDepth
				));
			}

			//
			// Set up shader binding table
			//
			OptixShaderBindingTable sbt = {};
			{
				CUdeviceptr  raygen_record;
				const size_t raygen_record_size = sizeof(RayGenSbtRecord);
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
				RayGenSbtRecord rg_sbt;
				OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
				CUDA_CHECK(cudaMemcpy(
					reinterpret_cast<void*>(raygen_record),
					&rg_sbt,
					raygen_record_size,
					cudaMemcpyHostToDevice
				));

				CUdeviceptr miss_record;
				size_t      miss_record_size = sizeof(MissSbtRecord);
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
				MissSbtRecord ms_sbt;
				ms_sbt.data = { 0.3f, 0.1f, 0.2f };
				OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
				CUDA_CHECK(cudaMemcpy(
					reinterpret_cast<void*>(miss_record),
					&ms_sbt,
					miss_record_size,
					cudaMemcpyHostToDevice
				));

				CUdeviceptr hitgroup_record;
				size_t      hitgroup_record_size = sizeof(HitGroupSbtRecord);
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
				HitGroupSbtRecord hg_sbt;
				OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
				CUDA_CHECK(cudaMemcpy(
					reinterpret_cast<void*>(hitgroup_record),
					&hg_sbt,
					hitgroup_record_size,
					cudaMemcpyHostToDevice
				));

				sbt.raygenRecord = raygen_record;
				sbt.missRecordBase = miss_record;
				sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
				sbt.missRecordCount = 1;
				sbt.hitgroupRecordBase = hitgroup_record;
				sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
				sbt.hitgroupRecordCount = 1;
			}

			//
			// launch
			//
			{
				CUstream stream;
				CUDA_CHECK(cudaStreamCreate(&stream));

				Camera cam;
				configureCamera(cam, width, height);

				Params params;
				params.image = outputBuffer;
				params.image_width = width;
				params.image_height = height;
				params.pitch = pitch;
				params.handle = gas_handle;
				params.cam_eye = cam.eye();
				cam.UVWFrame(params.cam_u, params.cam_v, params.cam_w);

				CUdeviceptr d_param;
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
				CUDA_CHECK(cudaMemcpy(
					reinterpret_cast<void*>(d_param),
					&params, sizeof(params),
					cudaMemcpyHostToDevice
				));

				OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, width, height, /*depth=*/1));
				CUDA_SYNC_CHECK();

				CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_param)));
			}

			//
			// Display results
			//
	   //     {
	   //         sutil::ImageBuffer buffer;
	   //         buffer.data         = output_buffer.getHostPointer();
	   //         buffer.width        = width;
	   //         buffer.height       = height;
	   //         buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

				//std::string outfile = "out.png";
				//sutil::saveImage( outfile.c_str(), buffer, false );
	   //     }
		}
	}

}
