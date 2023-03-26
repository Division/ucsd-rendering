
//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <optix.h>

#include <cuda/helpers.h>

#include <sutil/vec_math.h>

#include "Device/DeviceTypes.h"



struct MissData
{
    glm::vec3 bg_color;
};


extern "C" {
__constant__ Device::Params params;
}


static __forceinline__ __device__ void setPayload(const Device::RayPayload& payload)
{
    optixSetPayload_0(payload.raw[0]);
    optixSetPayload_1(payload.raw[1]);
    optixSetPayload_2(payload.raw[2]);
    optixSetPayload_3(payload.raw[3]);
    optixSetPayload_4(payload.raw[4]);
    optixSetPayload_5(payload.raw[5]);
    optixSetPayload_6(payload.raw[6]);
    optixSetPayload_7(payload.raw[7]);
    optixSetPayload_8(payload.raw[8]);
}


Device::RayPayload __device__ trace(OptixTraversableHandle handle, const Device::Ray& ray, bool terminateOnFirstHit = false)
{
    Device::RayPayload payload;

    uint32_t flags = OPTIX_RAY_FLAG_NONE;

    if (terminateOnFirstHit)
        flags |= OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT;

    optixTrace(
            handle,
            (float3&)ray.origin,
            (float3&)ray.direction,
            0.0f,                // Min intersection distance
            1000.0f,               // Max intersection distance
            0.0f,                // rayTime -- used for motion blur
            OptixVisibilityMask( 255 ), // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset   -- See SBT discussion
            1,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            payload.raw[0], payload.raw[1], payload.raw[2], payload.raw[3], payload.raw[4], payload.raw[5], payload.raw[6], payload.raw[7], payload.raw[8]);

    return payload;
}


static __forceinline__ __device__ Device::Ray computeRay( glm::uvec3 idx, glm::uvec3 dim)
{
    const glm::vec3 U = params.cam_u;
    const glm::vec3 V = params.cam_v;
    const glm::vec3 W = params.cam_w;
    const glm::vec2 d = 2.0f * glm::vec2(
            static_cast<float>( idx.x + 0.5 ) / static_cast<float>( dim.x ),
            static_cast<float>( idx.y + 0.5 ) / static_cast<float>( dim.y )) - 1.0f;

    return Device::Ray{ params.cam_eye, glm::normalize(d.x * U + d.y * V + W) };
}


Device::Ray __device__ appendRayEpsilon(const Device::Ray& ray)
{
    return Device::Ray{ ray.origin + ray.direction * 0.001f, ray.direction };
}

Device::Ray __device__ appendRayNormalEpsilon(const Device::Ray& ray, const glm::vec3& normal)
{
    return Device::Ray{ ray.origin + normal * 0.001f, ray.direction };
}

bool __device__ rayPayloadIsMiss(const Device::RayPayload& payload)
{
    return payload.values.instanceId == -1;
}


bool __device__ checkAnyHit(const Device::Ray& ray)
{
    auto result = trace(params.handle, ray, true);
    return !rayPayloadIsMiss(result);
}


glm::vec3 __device__ calculateLighting(const Device::Ray& initialRay)
{
    glm::vec3 colorSumm(0, 0, 0);

	glm::vec3 throughput = glm::vec3(1.0f);
	Device::Ray ray = initialRay;

    for (uint32_t bounceIndex = 0; bounceIndex < 6; bounceIndex++)
    {
        if (glm::dot(throughput, throughput) < 0.001f)
        {
            break;
        }

        // Trace the ray against our scene hierarchy
        Device::RayPayload payload = trace(params.handle, ray);


        if (rayPayloadIsMiss(payload))
        {
            break;
        }

        const Device::InstanceData& instanceData = params.instances[payload.values.instanceId];
        glm::vec3 color(0, 0, 0);
        color += instanceData.ambient + instanceData.emission;

        const glm::vec3 N = glm::normalize(payload.values.normal);
        const glm::vec3 V = glm::normalize(-ray.direction);

        for (uint32_t i = 0; i < params.directLightCount; i++)
        {
            const Device::Scene::DirectLight& directLight = params.directLights[i];
            const glm::vec3 L = glm::normalize(-directLight.direction);
            if (checkAnyHit(appendRayEpsilon({ payload.values.intersection, L })))
            {
                continue;
            }

            const glm::vec3 diffuse = instanceData.diffuse * glm::max(glm::dot(N, L), 0.0f);
            const glm::vec3 H = glm::normalize(L + V);
            const glm::vec3 specular = instanceData.specular * pow(glm::max(glm::dot(N, H), 0.0f), instanceData.shininess);
            const glm::vec3 lightColor = directLight.color * (diffuse + specular);
            color += lightColor;
        }

        for (uint32_t i = 0; i < params.pointLightCount; i++)
        {
            const Device::Scene::PointLight& pointLight = params.pointLights[i];
            glm::vec3 L = pointLight.position - payload.values.intersection;
            const float R2 = glm::dot(L, L);
            const float R = sqrt(R2);
            L /= R;

            if (checkAnyHit(appendRayNormalEpsilon({ payload.values.intersection, L }, N)))
            {
                continue;
            }

            const glm::vec3 diffuse = instanceData.diffuse * glm::max(glm::dot(N, L), 0.0f);
            const glm::vec3 H = glm::normalize(L + V);
            const glm::vec3 specular = instanceData.specular * pow(glm::max(glm::dot(N, H), 0.0f), instanceData.shininess);
            const glm::vec3 lightColor = pointLight.color * (diffuse + specular);
            const float attenuation = params.attenuation.x + R * params.attenuation.y + R2 * params.attenuation.z;
            color += lightColor / attenuation;
        }

        colorSumm += color * throughput;

        const glm::vec3 reflDir = glm::normalize(glm::reflect(ray.direction, N));
        ray = appendRayNormalEpsilon({ payload.values.intersection, reflDir }, N);
        throughput *= instanceData.specular;
    }

    return colorSumm;
}

 
extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const glm::uvec3 idx = (glm::uvec3&)optixGetLaunchIndex();
    const glm::uvec3 dim = (glm::uvec3&)optixGetLaunchDimensions();

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    const Device::Ray ray = computeRay(idx, dim);
	const glm::vec3 lighting = calculateLighting(ray);
    glm::vec4 result = glm::vec4(lighting, 1.0f);

    // Record results in our output raster
    float4* output = (float4*)(params.image + idx.x * sizeof(float4) + idx.y * params.pitch);
    *output = (float4&)result;
}


extern "C" __global__ void __miss__ms()
{
    MissData* miss_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );

    Device::RayPayload payload = {};
    payload.values.instanceId = -1;
    payload.values.primitiveId = -1;
    payload.values.normal = miss_data->bg_color;
    setPayload(payload);
}


extern "C" __global__ void __closesthit__sphere()
{
    const uint32_t instanceId = optixGetInstanceId();
    const Device::InstanceData& instance = params.instances[instanceId];
    const auto ambient = instance.ambient;
    //const auto ambient = glm::vec3(1,0,0);

    //setPayload( make_float3(0, 0, 1));

    float t_hit = optixGetRayTmax();
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const unsigned int           prim_idx    = optixGetPrimitiveIndex();
    const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
    const unsigned int           sbtGASIndex = optixGetSbtGASIndex();

    float4 q;
    // sphere center (q.x, q.y, q.z), sphere radius q.w
    optixGetSphereData( gas, prim_idx, sbtGASIndex, 0.f, &q );

    float3 world_raypos = ray_orig + t_hit * ray_dir;
    float3 obj_raypos   = optixTransformPointFromWorldToObjectSpace( world_raypos );
    float3 obj_normal   = ( obj_raypos - make_float3( q ) ) / q.w;
    float3 world_normal = normalize( optixTransformNormalFromObjectToWorldSpace( obj_normal ) );

    Device::RayPayload payload;
    payload.values.instanceId = instanceId;
    payload.values.primitiveId = prim_idx;
    payload.values.t = t_hit;
    payload.values.normal = (glm::vec3&)world_normal;
    payload.values.intersection = (glm::vec3&)world_raypos;
    setPayload(payload);
}


extern "C" __global__ void __closesthit__ch()
{
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const float barycentricsZ = 1.0f - barycentrics.x - barycentrics.y;

    const uint32_t instanceId = optixGetInstanceId();
    const Device::InstanceData& instance = params.instances[instanceId];
    const unsigned int primitiveIndex = optixGetPrimitiveIndex();
    const uint32_t normalOffset = params.instancesExtraData[instanceId].triangleNormalsOffset + primitiveIndex * 3;
    const glm::vec3 normal = barycentrics.x * params.triangleNormals[normalOffset] +
        barycentrics.y * params.triangleNormals[normalOffset + 1] +
        barycentricsZ * params.triangleNormals[normalOffset + 2];
    float3 normalWorldspace = normalize( optixTransformNormalFromObjectToWorldSpace( (float3&)normal ) );

    float t_hit = optixGetRayTmax();
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    float3 world_raypos = ray_orig + t_hit * ray_dir;

    Device::RayPayload payload;
    payload.values.instanceId = instanceId;
    payload.values.primitiveId = primitiveIndex;
    payload.values.t = t_hit;
    payload.values.normal = (glm::vec3&)normalWorldspace;
    payload.values.intersection = (glm::vec3&)world_raypos;
    setPayload(payload);
}

