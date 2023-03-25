#include "SceneParser.h"
#include "utils/FileLoader.h"

namespace Loader::Scene
{

	std::optional<TextScene> ParseTextScene(const std::wstring& path)
	{
		std::ifstream fileStream(path);

		std::string line;

		TextScene result;

		std::vector<glm::mat4> transformStack;
		transformStack.push_back(glm::identity<glm::mat4>());
		Instance currentInstance;

		while (std::getline(fileStream, line))
		{
			std::stringstream stream(line);
			std::string command;
			stream >> command;


			if (command.empty() || command[0] == '#')
			{
				continue;
			}

			if (command == "size")
			{
				stream >> result.size.x;
				stream >> result.size.y;
			}
			else if (command == "maxdepth")
			{
				stream >> result.maxBounces;
			}
			else if (command == "output")
			{
				stream >> result.outputPath;
			}
			else if (command == "ambient")
			{
				glm::vec3 c;
				stream >> c.x;
				stream >> c.y;
				stream >> c.z;
				currentInstance.ambient = c;
			}
			else if (command == "diffuse")
			{
				glm::vec3 c;
				stream >> c.x;
				stream >> c.y;
				stream >> c.z;
				currentInstance.diffuse = c;
			}
			else if (command == "specular")
			{
				glm::vec3 c;
				stream >> c.x;
				stream >> c.y;
				stream >> c.z;
				currentInstance.specular = c;
			}
			else if (command == "emission")
			{
				glm::vec3 c;
				stream >> c.x;
				stream >> c.y;
				stream >> c.z;
				currentInstance.emission = c;
			}
			else if (command == "shininess")
			{
				stream >> currentInstance.shininess;
			}
			else if (command == "camera")
			{
				stream >> result.camera.from.x;
				stream >> result.camera.from.y;
				stream >> result.camera.from.z;
				stream >> result.camera.to.x;
				stream >> result.camera.to.y;
				stream >> result.camera.to.z;
				stream >> result.camera.up.x;
				stream >> result.camera.up.y;
				stream >> result.camera.up.z;
				stream >> result.camera.fovY;
				result.camera.fovY = glm::radians(result.camera.fovY);
				result.camera.matrix = glm::lookAtRH(result.camera.from, result.camera.to, result.camera.up);
			}
			else if (command == "directional")
			{
				DirectLight light;
				stream >> light.direction.x;
				stream >> light.direction.y;
				stream >> light.direction.z;
				stream >> light.color.x;
				stream >> light.color.y;
				stream >> light.color.z;
				result.directLights.push_back(light);
			}
			else if (command == "point")
			{
				PointLight light;
				stream >> light.position.x;
				stream >> light.position.y;
				stream >> light.position.z;
				stream >> light.color.x;
				stream >> light.color.y;
				stream >> light.color.z;
				result.pointLights.push_back(light);
			}
			else if (command == "attenuation")
			{
				stream >> result.constAttenuation;
				stream >> result.linearAttenuation;
				stream >> result.quadraticAttenuation;
			}
			else if (command == "sphere")
			{
				Sphere sphere;

				stream >> sphere.position.x;
				stream >> sphere.position.y;
				stream >> sphere.position.z;
				stream >> sphere.radius;
				currentInstance.spheres.push_back(sphere);
			}
			else if (command == "vertex")
			{
				glm::vec3 v;
				stream >> v.x;
				stream >> v.y;
				stream >> v.z;
				result.vertices.push_back(v);
			}
			else if (command == "tri")
			{
				Triangle t;
				stream >> t.v0;
				stream >> t.v1;
				stream >> t.v2;
				currentInstance.triangles.push_back(t);

				glm::vec3 v0 = result.vertices[t.v0];
				glm::vec3 v1 = result.vertices[t.v1];
				glm::vec3 v2 = result.vertices[t.v2];

				const glm::vec3 normal = glm::normalize(glm::cross(v2 - v1, v0 - v1));
				currentInstance.normals.push_back(normal);
				currentInstance.normals.push_back(normal);
				currentInstance.normals.push_back(normal);
			}
			else if (command == "maxverts")
			{
			}
			else if (command == "trinormal")
			{
			}
			else if (command == "pushTransform")
			{
				transformStack.push_back(currentInstance.transform);
			}
			else if (command == "popTransform")
			{
				if (transformStack.size() <= 1)
				{
					std::cout << "popTransform mismatch\n";
					return std::nullopt;
				}

				result.instances.push_back(std::move(currentInstance));
				currentInstance.transform = transformStack.back();
				transformStack.pop_back();
			}
			else if (command == "translate")
			{
				glm::vec3 t;
				stream >> t.x;
				stream >> t.y;
				stream >> t.z;
				currentInstance.transform = glm::translate(currentInstance.transform, t);
			}
			else if (command == "rotate")
			{
				glm::vec3 axis;
				float angleDegrees;
				stream >> axis.x;
				stream >> axis.y;
				stream >> axis.z;
				stream >> angleDegrees;
				currentInstance.transform = glm::rotate(currentInstance.transform, glm::radians(angleDegrees), axis);
			}
			else if (command == "scale")
			{
				glm::vec3 s;
				stream >> s.x;
				stream >> s.y;
				stream >> s.z;
				currentInstance.transform = glm::scale(currentInstance.transform, s);
			}
			else
			{
				throw std::runtime_error("Unknown command");
			}

			if (!stream)
			{
				return std::nullopt;
			}
		}
		
		if (result.instances.empty())
		{
			result.instances.push_back(currentInstance);
		}

		return result;
	}

}
