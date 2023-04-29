#pragma once

#include "Device/DeviceTypes.h"

namespace Loader::Scene
{
	enum class IntegratorType
	{
		Default,
		AnalyticDirect,
		Direct,
		PathTracer
	};
	
	struct Camera
	{
		glm::vec3 from;
		glm::vec3 to;
		glm::vec3 up;
		float fovY;
		glm::mat4 matrix;
	};

	struct Triangle
	{
		uint32_t v0;
		uint32_t v1;
		uint32_t v2;
	};

	typedef Device::Scene::Sphere Sphere;
	typedef Device::Scene::DirectLight DirectLight;
	typedef Device::Scene::PointLight PointLight;
	typedef Device::Scene::QuadLight QuadLight;

	struct Instance : public Device::InstanceData
	{
		glm::mat4 transform = glm::identity<glm::mat4>();
		std::vector<Sphere> spheres;
		std::vector<Triangle> triangles;
		std::vector<glm::vec3> normals;
	};

	class TextScene
	{
		friend std::optional<TextScene> ParseTextScene(const std::wstring& path);

	public:
		glm::uvec2 size;
		std::string outputPath;
		uint32_t spp = 1;
		uint32_t maxBounces = 5;
		uint32_t lightSamples = 1;
		bool lightStratify = false;
		Camera camera;
		glm::vec3 attenuation = { 1, 0, 0 }; // x const, y linear, z quadratic
		std::vector<DirectLight> directLights;
		std::vector<PointLight> pointLights;
		std::vector<QuadLight> quadLights;
		std::vector<Instance> instances;
		std::vector<glm::vec3> vertices;
		IntegratorType integratorType = IntegratorType::Default;

		float constAttenuation = 1;
		float linearAttenuation = 0;
		float quadraticAttenuation = 0;

	public:

	};

	std::optional<TextScene> ParseTextScene(const std::wstring& path);


}
