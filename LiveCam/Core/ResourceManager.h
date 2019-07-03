#pragma once

#include <unordered_map>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>

#include <Common/CommonClasses.h>

namespace fs = boost::filesystem;

class ResourceManager
{
public:
	boost::filesystem::path resourceRoot = "";

	struct TextureLoadingResult
	{
		GLuint ID;
		size_t textureWidth;
		size_t textureHeight;
	};

	struct AnimationLoadingResult
	{
		std::vector<GLuint> IDs;
		size_t animationWidth;
		size_t animationHeight;
		size_t animationLength;
	};

	struct AtlasLoadingResult
	{
		struct AtlasTexture
		{
			Eigen::Vector2f shift;
		};

		GLuint ID;
		size_t atlasWidth;
		size_t atlasHeight;
		Eigen::Vector2f atlasTextureSize;
		size_t count;
		std::vector<AtlasTexture> atlasTextures;
	};

	static void createTexture(cv::Mat image);
	static GLuint createCubemap(std::array<std::string, 6> paths);

	TextureLoadingResult loadTexture(std::string &texture);
	TextureLoadingResult loadTexture(std::string &texture, cv::Mat image);
	AnimationLoadingResult loadAnimation(std::string &folderpath);
	AtlasLoadingResult loadAtlas(std::string &atlaspath);
	GLuint loadCubemap(std::array<std::string, 6> &cubemapTextures);

	void removeTextures(std::vector<std::string> &textures);
	void removeAnimation(std::string &animationPath);
	void removeCubemap(std::array<std::string, 6> &cubemapTextures);
	void clear();

private:
	std::unordered_map<std::string, TextureLoadingResult> cachedTextures;
	std::unordered_map<std::string, AnimationLoadingResult> cachedAnimations;
	std::unordered_map<std::string, AtlasLoadingResult> cachedAtlases;
	std::unordered_map<std::string, GLuint> cachedCubemaps;
};
