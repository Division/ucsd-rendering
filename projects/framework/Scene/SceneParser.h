#pragma once

namespace Loader::Scene
{

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

	struct Instance
	{
		glm::mat4 transform = glm::identity<glm::mat4>();
		glm::vec3 diffuse = {1, 1, 1};
		glm::vec3 specular = {1, 1, 1};
		glm::vec3 shininess = {1, 1, 1};
		glm::vec3 emission = {1, 1, 1};
		glm::vec3 ambient = { 0.2f, 0.2f, 0.2f };

		std::vector<Sphere> spheres;
		std::vector<Triangle> triangles;


	};

	class TextScene
	{
		friend std::optional<TextScene> ParseTextScene(const std::wstring& path);

	public:
		glm::uvec2 size;
		std::string outputPath;
		uint32_t maxBounces = 5;
		Camera camera;
		glm::vec3 attenuation = { 1, 0, 0 }; // x const, y linear, z quadratic
		std::vector<DirectLight> directLights;
		std::vector<PointLight> pointLights;
		std::vector<Instance> instances;
		std::vector<glm::vec3> vertices;
		std::vector<glm::vec3> normals;

	public:

	};

	std::optional<TextScene> ParseTextScene(const std::wstring& path);


}
