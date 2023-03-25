#define STB_IMAGE_WRITE_IMPLEMENTATION
#pragma warning(disable : 4996)
#include "stb_image_write.h"

namespace Image
{
	bool SaveAsPNG(const void* data, const uint32_t w, const uint32_t h, const std::wstring& path)
	{
		auto write_func = [](void* context, void* data, int size)
		{
			auto* stream = static_cast<std::ofstream*>(context);
			stream->write((char*)data, size);
		};

		std::ofstream stream(path, std::ios::binary);
		if (!stream)
			return false;

		const bool result = stbi_write_png_to_func(write_func, &stream, w, h, 4, data, 0);
		return result && stream;
	}
}