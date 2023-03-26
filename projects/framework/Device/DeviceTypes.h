#pragma once

#include <optix.h>
#include <glm/glm.hpp>
#include <array>

namespace Device
{
	struct Ray
	{
		glm::vec3 origin;
		glm::vec3 direction;
	};

	namespace Scene
	{
		struct Sphere
		{
			glm::vec3 position;
			float radius;
		};

		struct DirectLight
		{
			glm::vec3 direction;
			glm::vec3 color;
		};

		struct PointLight
		{
			glm::vec3 position;
			glm::vec3 color;
		};

	}

	struct alignas(glm::vec4) InstanceData
	{
		glm::vec3 diffuse = {1, 1, 1};
		glm::vec3 specular = {1, 1, 1};
		float shininess = 0;
		glm::vec3 emission = {0, 0, 0};
		glm::vec3 ambient = { 0.2f, 0.2f, 0.2f };
	};

	struct alignas(glm::vec4) InstanceExtraData
	{
		uint32_t triangleNormalsOffset;
	};

	struct Params
	{
		uint8_t* image;
		const InstanceData* instances;
		const InstanceExtraData* instancesExtraData;
		const Scene::DirectLight* directLights;
		const Scene::PointLight* pointLights;
		const glm::vec3* triangleNormals;
		unsigned int image_width;
		unsigned int image_height;
		unsigned int pitch;
		glm::vec3 cam_eye;
		glm::vec3 cam_u, cam_v, cam_w;
		OptixTraversableHandle handle;
		uint32_t directLightCount;
		uint32_t pointLightCount;
		glm::vec3 attenuation;
	};


	struct RayPayload
	{
		union {
			struct alignas(uint32_t) Values {
				float t;
				glm::vec3 intersection;
				glm::vec3 normal;
				uint32_t instanceId;
				uint32_t primitiveId;
			} values;

			uint32_t raw[sizeof(Values) / sizeof(uint32_t)];
		};

		static constexpr uint32_t GetPayloadSize() { return sizeof(RayPayload) / sizeof(uint32_t); }
	};

}
