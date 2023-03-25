#include <optix.h>
#include <optix_stack_size.h>
#include <sutil/Trackball.h>
#include "utils/FileLoader.h"
#include "utils/CUDAHelper.h"
#include <sutil/vec_math.h>
#include "utils/OptixHelper.h"
#include <optix_stubs.h>
#include "Scene/SceneParser.h"


using namespace Optix;

namespace
{
	class Camera {
public:
    Camera()
        : m_eye(1.0f), m_lookat(0.0f), m_up(0.0f, 1.0f, 0.0f), m_fovY(glm::radians(35.0f)), m_aspectRatio(1.0f)
    {
    }

    Camera(const glm::vec3& eye, const glm::vec3& lookat, const glm::vec3& up, float fovY, float aspectRatio)
        : m_eye(eye), m_lookat(lookat), m_up(up), m_fovY(fovY), m_aspectRatio(aspectRatio)
    {
    }

    glm::vec3 direction() const { return normalize(m_lookat - m_eye); }
    void setDirection(const glm::vec3& dir) { m_lookat = m_eye + length(m_lookat - m_eye) * dir; }

    const glm::vec3& eye() const { return m_eye; }
    void setEye(const glm::vec3& val) { m_eye = val; }
    const glm::vec3& lookat() const { return m_lookat; }
    void setLookat(const glm::vec3& val) { m_lookat = val; }
    const glm::vec3& up() const { return m_up; }
    void setUp(const glm::vec3& val) { m_up = val; }
    const float& fovY() const { return m_fovY; }
    void setFovY(const float& val) { m_fovY = val; }
    const float& aspectRatio() const { return m_aspectRatio; }
    void setAspectRatio(const float& val) { m_aspectRatio = val; }

    // UVW forms an orthogonal, but not orthonormal basis!
	void UVWFrame(glm::vec3& U, glm::vec3& V, glm::vec3& W) const
	{
		W = m_lookat - m_eye; // Do not normalize W -- it implies focal length
		float wlen = length(W);
		U = normalize(cross(W, m_up));
		V = normalize(cross(U, W));

		float vlen = wlen * tanf(0.5f * m_fovY);
		V *= vlen;
		float ulen = vlen * m_aspectRatio;
		U *= ulen;
	}

private:
    glm::vec3 m_eye;
    glm::vec3 m_lookat;
    glm::vec3 m_up;
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


	typedef SbtRecord<RayGenData>     RayGenSbtRecord;
	typedef SbtRecord<MissData>       MissSbtRecord;
	typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;

	void configureCamera(Camera& cam, const Loader::Scene::Camera& sceneCamera, const uint32_t width, const uint32_t height)
	{
		cam.setEye(sceneCamera.from);
		cam.setLookat(sceneCamera.to);
		cam.setUp(sceneCamera.up);
		cam.setFovY(sceneCamera.fovY);
		cam.setAspectRatio((float)width / (float)height);
	}

	struct SceneRenderData
	{
		std::vector<std::unique_ptr<Acceleration::Structure>> instancesAS;
		std::unique_ptr<Acceleration::Structure> toplevelAS;
		std::unique_ptr<CUDA::DeviceMemory> instanceData;
		std::unique_ptr<CUDA::DeviceMemory> instanceExtraData;
		std::unique_ptr<CUDA::DeviceMemory> triangleNormals;
	};

	SceneRenderData GetSceneRenderData(const Loader::Scene::TextScene& scene)
	{
		SceneRenderData result;

		std::vector<glm::vec3> vertices;
		std::vector<Acceleration::Init::Instance> instances;
		std::vector<Device::InstanceData> gpuInstanceData;
		std::vector<glm::vec3> gpuTriangleNormals;
		std::vector<Device::InstanceExtraData> gpuExtraInstanceData;

		for (auto& instance : scene.instances)
		{
			vertices.reserve(instance.triangles.size() * 3);
			vertices.clear();
			for (auto& t : instance.triangles)
			{
				vertices.push_back(scene.vertices[t.v0]);
				vertices.push_back(scene.vertices[t.v1]);
				vertices.push_back(scene.vertices[t.v2]);
			}

			if (vertices.size())
			{
				const auto triangles = Acceleration::Init::Triangles(vertices);
				result.instancesAS.push_back(std::make_unique<Acceleration::Structure>(triangles));
				instances.push_back(Acceleration::Init::Instance{ .transform = instance.transform, .structure = result.instancesAS.back().get() });
				gpuInstanceData.push_back(instance);

				Device::InstanceExtraData extraData{ .triangleNormalsOffset = static_cast<uint32_t>(gpuTriangleNormals.size()) };
				gpuExtraInstanceData.push_back(extraData);
				gpuTriangleNormals.insert(gpuTriangleNormals.end(), instance.normals.begin(), instance.normals.end());
			}

			if (instance.spheres.size())
			{
				const auto spheres = Acceleration::Init::Spheres(instance.spheres);
				result.instancesAS.push_back(std::make_unique<Acceleration::Structure>(spheres));
				instances.push_back(Acceleration::Init::Instance{ .transform = instance.transform, .structure = result.instancesAS.back().get(), .sbtOffset = 1 });
				gpuInstanceData.push_back(instance);

				Device::InstanceExtraData extraData{};
				gpuExtraInstanceData.push_back(extraData);
			}
		}

		const auto toplevel = Acceleration::Init::Instances(instances);
		result.toplevelAS = std::make_unique<Acceleration::Structure>(toplevel);
		result.instanceData = std::make_unique<CUDA::DeviceMemory>(gsl::span<const Device::InstanceData>(gpuInstanceData));
		result.instanceExtraData = std::make_unique<CUDA::DeviceMemory>(gsl::span<const Device::InstanceExtraData>(gpuExtraInstanceData));
		result.triangleNormals = std::make_unique<CUDA::DeviceMemory>(gsl::span<const glm::vec3>(gpuTriangleNormals));

		return result;
	}


	void SetupOptix(const uint32_t width, const uint32_t height, const uint32_t pitch, void* outputBuffer)
	{
		if (!Initialize())
		{
			std::cerr << "Failed to initialize optix\n";
			throw std::runtime_error("Failed to initialize optix");
		}

		auto scene = Loader::Scene::ParseTextScene(L"data/homework1/testscenes/scene3.test");
		if (!scene)
		{
			throw std::runtime_error("Failed to load scene");
		}

		SceneRenderData renderData = GetSceneRenderData(*scene);

		OptixDeviceContext context = GetContext();
		optixDeviceContextSetCacheEnabled(context, 0);
		//
		// Create program groups
		//

		ModuleInit moduleInit(L"data/kernel/recursiveRayTracing.cu.obj", Device::RayPayload::GetPayloadSize(), 3, OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE);
		const auto raygenIdx = moduleInit.AddRaygenProgram("__raygen__rg");
		const auto missIdx = moduleInit.AddMissProgram("__miss__ms");
		const auto hitIdx = moduleInit.AddHitProgram("__closesthit__ch", "", "", false);
		const auto hitSphereIdx = moduleInit.AddHitProgram("__closesthit__sphere", "", "", true);

		Module module(moduleInit);

		//
		// Link pipeline
		//

		PipelineInit pipelineInit(module, 3);
		pipelineInit.AddSbtValue(OPTIX_PROGRAM_GROUP_KIND_MISS, missIdx, MissSbtRecord{ .data = { 0.3f, 0.1f, 0.2f } });
		pipelineInit.AddSbtValue(OPTIX_PROGRAM_GROUP_KIND_RAYGEN, raygenIdx, EmptySbtRecord{});
		pipelineInit.AddSbtValue(OPTIX_PROGRAM_GROUP_KIND_HITGROUP, hitIdx, EmptySbtRecord{});
		pipelineInit.AddSbtValue(OPTIX_PROGRAM_GROUP_KIND_HITGROUP, hitSphereIdx, EmptySbtRecord{});
		Pipeline pipeline(pipelineInit);

		//
		// launch
		//
		{
			Camera cam;
			configureCamera(cam, scene->camera, width, height);

			Device::Params params;
			params.image = static_cast<uint8_t*>(outputBuffer);
			params.instances = static_cast<const Device::InstanceData*>(renderData.instanceData->GetMemory());
			params.instancesExtraData = static_cast<const Device::InstanceExtraData*>(renderData.instanceExtraData->GetMemory());
			params.triangleNormals = static_cast<const glm::vec3*>(renderData.triangleNormals->GetMemory());
			params.image_width = width;
			params.image_height = height;
			params.pitch = pitch;
			params.handle = renderData.toplevelAS->GetHandle();
			params.cam_eye = cam.eye();
			cam.UVWFrame(params.cam_u, params.cam_v, params.cam_w);

			Launch(params, pipeline, width, height);
		}
	}

}
