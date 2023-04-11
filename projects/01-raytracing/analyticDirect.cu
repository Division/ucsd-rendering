
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
#include "glm/gtx/intersect.hpp"



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


Device::RayPayload __device__ trace(OptixTraversableHandle handle, const Device::Ray& ray, float minDist = 0.0f, float maxDist = 1000000.0f, bool terminateOnFirstHit = false)
{
    Device::RayPayload payload;

    uint32_t flags = OPTIX_RAY_FLAG_NONE;

    if (terminateOnFirstHit)
        flags |= OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT;

    optixTrace(
            handle,
            (float3&)ray.origin,
            (float3&)ray.direction,
            minDist,             // Min intersection distance
            maxDist,             // Max intersection distance
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


bool __device__ checkAnyHit(const Device::Ray& ray, float maxDist = 1000000.0f)
{
    auto result = trace(params.handle, ray, 0.0f, maxDist, true);
    return !rayPayloadIsMiss(result);
}


glm::vec3 __device__ calculateLighting(const Device::Ray& initialRay)
{
	glm::vec3 colorSumm(0, 0, 0);

	const Device::Ray ray = initialRay;

    for (uint32_t i = 0; i < params.sceneData->quadLightCount; i++)
    {
        const Device::Scene::QuadLight& quadLight = params.sceneData->quadLights[i];
        glm::vec3 v0, v1, v2, v3;
        v0 = quadLight.origin;
        v1 = quadLight.origin + quadLight.va;
        v2 = quadLight.origin + quadLight.vb;
        v3 = quadLight.origin + quadLight.va + quadLight.vb;

        glm::vec2 bary;
        float dist;
        if (glm::intersectRayTriangle(ray.origin, ray.direction, v0, v1, v2, bary, dist) ||
            glm::intersectRayTriangle(ray.origin, ray.direction, v1, v2, v3, bary, dist))
        {
            return quadLight.color;
        }
    }

	// Trace the ray against our scene hierarchy
	Device::RayPayload payload = trace(params.handle, ray);

	if (rayPayloadIsMiss(payload))
	{
		return colorSumm;
	}

	const Device::InstanceData& instanceData = params.sceneData->instances[payload.values.instanceId];
	glm::vec3 color(0, 0, 0);
	//color += instanceData.ambient + instanceData.emission;

	const glm::vec3 N = glm::normalize(payload.values.normal);
	const glm::vec3 r = payload.values.intersection;

	for (uint32_t i = 0; i < params.sceneData->quadLightCount; i++)
	{
		const Device::Scene::QuadLight& quadLight = params.sceneData->quadLights[i];
		glm::vec3 v0, v1, v2, v3;
		v0 = quadLight.origin;
		v1 = quadLight.origin + quadLight.va;
		v2 = quadLight.origin + quadLight.va + quadLight.vb;
		v3 = quadLight.origin + quadLight.vb;

		glm::vec3 u0, u1, u2, u3;
		u0 = glm::normalize(v0 - r);
		u1 = glm::normalize(v1 - r);
		u2 = glm::normalize(v2 - r);
		u3 = glm::normalize(v3 - r);

		float O0, O1, O2, O3;
		O0 = acosf(glm::dot(u0, u1));
		O1 = acosf(glm::dot(u1, u2));
		O2 = acosf(glm::dot(u2, u3));
		O3 = acosf(glm::dot(u3, u0));

		glm::vec3 G0, G1, G2, G3;
		G0 = glm::normalize(glm::cross(u0, u1));
		G1 = glm::normalize(glm::cross(u1, u2));
		G2 = glm::normalize(glm::cross(u2, u3));
		G3 = glm::normalize(glm::cross(u3, u0));

		glm::vec3 Fi = 0.5f * (O0 * G0 + O1 * G1 + O2 * G2 + O3 * G3);

		const glm::vec3 direct = instanceData.diffuse / M_PIf * quadLight.color * glm::max(glm::dot(N, (Fi)), 0.0f);
		//const glm::vec3 direct = instanceData.diffuse / M_PIf * quadLight.color * glm::dot(N, (Fi));
		color += direct;

  //      Fi = glm::normalize((v0 + v1 + v2 + v3) / 4.0f - r);
		//color = glm::vec3(glm::dot(N, (Fi))) * 0.1f;
        //color = (glm::normalize(Fi) + glm::vec3(1.0f)) * 0.5f;
        //color = glm::normalize(Fi);// +glm::vec3(1.0f)) * 0.5f;
        //color = glm::vec3(glm::length(Fi));
	}

	colorSumm += color;

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
    const Device::InstanceData& instance = params.sceneData->instances[instanceId];
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
    const Device::InstanceData& instance = params.sceneData->instances[instanceId];
    const unsigned int primitiveIndex = optixGetPrimitiveIndex();
    const uint32_t normalOffset = params.sceneData->instancesExtraData[instanceId].triangleNormalsOffset + primitiveIndex * 3;
    const glm::vec3 normal = barycentrics.x * params.sceneData->triangleNormals[normalOffset] +
        barycentrics.y * params.sceneData->triangleNormals[normalOffset + 1] +
        barycentricsZ * params.sceneData->triangleNormals[normalOffset + 2];
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

