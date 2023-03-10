#include "Math.h"
#include "GPUScene.h"

using namespace Math;
using namespace RayTracing;

#define PI 3.1415926536f

typedef vec4 RGBA;

struct RenderData
{
    unsigned char* surface;
    unsigned char* surface_last_frame;
    int width; 
    int height;
    size_t pitch;
    int frame_index;
    GPUScene scene;
    cudaTextureObject_t env_tex;
};

struct HitData
{
    vec3 position;
    vec3 normal;
    vec2 uv = vec2(0);
    float distance = 1e30f;
    
    GPUMaterial material;
};

__device__ bool BVHRayHit(const Ray& ray, const GPUScene& scene, const GPUBVHNode* bvh, HitData& result)
{
    constexpr uint32_t node_stack_size = 64;
    uint32_t stack[node_stack_size];
    uint32_t stack_position = 0;
    stack[stack_position++] = 0; // Adding root node
    auto normalized_ray_direction = glm::normalize(ray.direction);
    
    const float original_maximum_distance = result.distance;

    while (stack_position)
    {
        auto& node = bvh[stack[--stack_position]];
        if (!IntersectAABB(ray, node.bounds, result.distance))
            continue;

        if (node.IsLeaf())
        {
            for (uint32_t i = 0; i < node.prim_count; i++)
            {
                auto& face = scene.gpu_faces[scene.gpu_bvh_face_indices[node.first_index + i]];
                auto& v0 = scene.gpu_vertices[face.v0];
                auto& v1 = scene.gpu_vertices[face.v1];
                auto& v2 = scene.gpu_vertices[face.v2];

                float distance;
                vec2 bary_coord;
                if (glm::intersectRayTriangle(ray.origin, normalized_ray_direction, v0.position, v1.position, v2.position, bary_coord, distance))
                {
                    if (distance >= result.distance || distance < 0.0f) continue;
                    const float bary_coord_z = 1.0f - bary_coord.x - bary_coord.y;
                    result.distance = distance;
                    result.position = ray.origin + normalized_ray_direction * distance;
                    result.normal = glm::normalize(v0.normal * bary_coord.x + v1.normal * bary_coord.y + v2.normal * bary_coord_z);
                    result.material = *scene.GetMaterial(face.material);
                    const bool backface = glm::dot(normalized_ray_direction, result.normal) >= 0.0f;
                    if (backface) result.normal = -result.normal;
                }
            }
        }
        else
        {
            stack[stack_position++] = node.first_index;
            stack[stack_position++] = node.first_index + 1;
        }
    }

    return result.distance < original_maximum_distance;
}

__device__ bool GetRayHit(const Ray& ray, GPUScene& scene, HitData& result)
{
    constexpr float max_distance = 1e30f;
    const auto normalized_ray_direction = glm::normalize(ray.direction);
    result.distance = max_distance;

    const int sphere_count = scene.GetSphereCount();
    for (int i = 0; i < sphere_count; i++)
    {
        auto sphere = scene.GetSphereData(i);
        float distance;
        if (glm::intersectRaySphere(ray.origin, normalized_ray_direction, sphere->position, sphere->radius * sphere->radius, distance))
        {
            if (distance >= result.distance) continue;
            result.distance = distance;
            result.position = ray.origin + normalized_ray_direction * distance;
            result.normal = (result.position - sphere->position) / sphere->radius;
            result.material = *scene.GetMaterial(sphere->material);
            result.distance  = distance;
        }
    }

    if (BVHRayHit(ray, scene, scene.gpu_bvh_nodes, result))
        result.distance = result.distance;

    return result.distance < max_distance;
}

__device__ RGBA ray_color(const Ray& r, GPUScene& scene, RNG rng) 
{
    vec3 result_color(0);
    vec3 throughput(1);
    constexpr int num_bounces = 6;

    Ray current_ray = r;
    for (int i = 0; i < num_bounces; i++)
    {
        HitData hit;
        if (GetRayHit(current_ray, scene, hit))
        {
            // calculate whether we are going to do a diffuse or specular reflection ray %
            const float do_specular = (rng.GetFloat01() < hit.material.specular_percent) ? 1.0f : 0.0f;

            result_color += throughput * vec3(hit.material.emissive);
			throughput *= vec3(glm::lerp(hit.material.albedo, hit.material.specular, do_specular));

            const vec3 diffuse_ray_dir = glm::normalize(hit.normal + rng.GetRandomPointOnSphere());
            vec3 specular_ray_dir = glm::normalize(glm::reflect(current_ray.direction, hit.normal));

            specular_ray_dir = glm::normalize(lerp(specular_ray_dir, diffuse_ray_dir, hit.material.roughness * hit.material.roughness));
            vec3 new_ray_dir = glm::normalize(glm::lerp(diffuse_ray_dir, specular_ray_dir, vec3(do_specular)));

            current_ray = Ray(hit.position + hit.normal * 0.01f, new_ray_dir);

            // Russian Roulette
			// As the throughput gets smaller, the ray is more likely to get terminated early.
			// Survivors have their value boosted to make up for fewer samples being in the average.
            {
                float p = glm::max(throughput.r, glm::max(throughput.g, throughput.b));
                if (rng.GetFloat01() > p)
                    break;

                // Add the energy we 'lose' by randomly terminating paths
                throughput *= 1.0f / p;
            }
        }
        else
        {
            auto rotated_direction = quat(vec3(0, PI, 0)) * current_ray.direction;
            float4 cubemap = texCubemapLod<float4>(scene.environment_cubemap_tex, rotated_direction.x, rotated_direction.y, rotated_direction.z, 0);
            result_color += throughput * /*glm::saturate*/glm::clamp(vec3(cubemap.x, cubemap.y, cubemap.z), vec3(0), vec3(50));

            break;
        }
    }

    return RGBA(result_color, 1);
}

__global__ void raytracing_kernel_main(RenderData render_data) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

#ifndef _DEBUG
    constexpr int sample_count = 5;
#else
	constexpr int sample_count = 1;
#endif

    // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't
    // correspond to valid pixels
    if (x >= render_data.width || y >= render_data.height) return;

    // get a pointer to the pixel at (x,y)
    RGBA* pixel = (RGBA*)(render_data.surface + y * render_data.pitch) + x;
    RGBA* pixel_last_frame = (RGBA*)(render_data.surface_last_frame + y * render_data.pitch) + x;

    *pixel = *pixel_last_frame;
    return;

    // populate it
    const vec2 pixel_coords(x, y);

    RNG rng = render_data.scene.GetRNGForPixel(render_data.width, x, y);

    vec4 result_color(0);

    for (int i = 0; i < sample_count; i++)
    {
		const auto uv = (pixel_coords + vec2(rng.GetFloat01(), rng.GetFloat01())) / vec2(render_data.width, render_data.height);
        Ray ray = render_data.scene.camera.GetRay(uv);
        result_color += ray_color(ray, render_data.scene, rng);
	}

    result_color /= (float)sample_count;
    float lerp_value = render_data.frame_index > 0 ? 1.0f / (float)(render_data.frame_index + 1) : 1.0f;
    *pixel = glm::lerp(*pixel_last_frame, result_color, lerp_value);

    pixel->a = 1.0f;
}

extern "C" void raytracing_process(void* surface, void* surface_last_frame, int width, int height, size_t pitch, int frame_index, GPUScene* scene) {
    cudaError_t error = cudaSuccess;

    dim3 Db = dim3(16, 16);  // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

    RenderData render_data{
        static_cast<unsigned char*>(surface), static_cast<unsigned char*>(surface_last_frame), width, height, pitch, 
        frame_index, *scene
    };

    raytracing_kernel_main<<<Dg, Db>>>(render_data);

    error = cudaGetLastError();

    if (error != cudaSuccess) {
        printf("(raytracing_kernel_main) failed to launch error = %d\n", error);
    }
}

