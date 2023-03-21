#pragma once

#include <optix.h>
#include <glm/glm.hpp>

namespace Device
{

	namespace Geometry
	{
		struct Sphere
		{
			glm::vec3 position;
			float radius;
		};
	}

	struct InstanceData
	{
		glm::vec3 diffuse = {1, 1, 1};
		glm::vec3 specular = {1, 1, 1};
		glm::vec3 shininess = {1, 1, 1};
		glm::vec3 emission = {1, 1, 1};
		glm::vec3 ambient = { 0.2f, 0.2f, 0.2f };
	};

	struct Params
	{
		uint8_t* image;
		const InstanceData* instances;
		unsigned int image_width;
		unsigned int image_height;
		unsigned int pitch;
		glm::vec3 cam_eye;
		glm::vec3 cam_u, cam_v, cam_w;
		OptixTraversableHandle handle;
	};

}
