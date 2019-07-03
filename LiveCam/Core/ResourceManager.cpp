#include <ResourceManager.h>

#include <Common/JSONVectorReader.h>

void ResourceManager::createTexture(cv::Mat image) {

	char * buffer = new char[image.rows*image.cols*image.channels()];
	int step = image.step;
	int height = image.rows;
	int width = image.cols;
	int channels = image.channels();
	char * data = (char *)image.data;

	for (int i = 0; i < height; i++)
	{
		memcpy(&buffer[i*width*channels], &(data[i*step]), width*channels);
	}
	if (channels == 4)
	{
		glTexImage2D(
			GL_TEXTURE_2D,
			0,
			GL_RGBA,
			image.cols,
			image.rows,
			0,
			GL_BGRA_EXT,
			GL_UNSIGNED_BYTE,
			buffer);
	}
	else
	{
		glTexImage2D(
			GL_TEXTURE_2D,
			0,
			GL_RGB,
			image.cols,
			image.rows,
			0,
			GL_BGR_EXT,
			GL_UNSIGNED_BYTE,
			buffer);
	}

	delete[] buffer;
}

GLuint ResourceManager::createCubemap(std::array<std::string, 6> paths)
{
	cv::Mat images[6];

	int width = 0;
	int height = 0;

	for (size_t i = 0; i < 6; i++)
	{
		images[i] = cv::imread(paths[i], cv::IMREAD_UNCHANGED);

		if (i == 0)
		{
			width = images[i].cols;
			height = images[i].rows;
		}
		else
		{
			if (images[i].cols != width || images[i].rows != height)
			{
				std::cerr << "Texture in cubemap have different size: " << paths[i] << std::endl;
				return -1;
			}
		}

		if (images[i].empty()) {
			std::cerr << "Unable to load texture: " << paths[i] << std::endl;
			return -1;
		}

		if (images[i].channels() != 3)
		{
			std::cerr << "Texture should not contain alpha channel: " << paths[i] << std::endl;
			return -1;
		}
	}

	GLuint id;

	glGenTextures(1, &id);
	glBindTexture(GL_TEXTURE_CUBE_MAP, id);

#ifdef _WIN32

	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

#endif

	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 0, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, images[0].data);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 1, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, images[1].data);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 2, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, images[2].data);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 3, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, images[3].data);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 4, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, images[4].data);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + 5, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, images[5].data);

	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

	return id;
}

ResourceManager::TextureLoadingResult ResourceManager::loadTexture(std::string &texture)
{
	auto iter = cachedTextures.find(texture);

	if (iter != cachedTextures.end())
	{
		return iter->second;
	}

	TextureLoadingResult result  = { 0, 0, 0 };

	auto image = cv::imread(fs::path(resourceRoot / texture).string(), cv::IMREAD_UNCHANGED);

	if (image.empty())
	{
		return result;
	}

	result.textureWidth = image.cols;
	result.textureHeight = image.rows;

	glGenTextures(1, &result.ID);
	glBindTexture(GL_TEXTURE_2D, result.ID);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
 	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	ResourceManager::createTexture(image);

	cachedTextures[texture] = result;

	return result;
}

ResourceManager::TextureLoadingResult ResourceManager::loadTexture(std::string &texture, cv::Mat image)
{
	auto iter = cachedTextures.find(texture);

	if (iter != cachedTextures.end())
	{
		return iter->second;
	}

	TextureLoadingResult result = { 0, 0, 0 };

	result.textureWidth = image.cols;
	result.textureHeight = image.rows;

	glGenTextures(1, &result.ID);
	glBindTexture(GL_TEXTURE_2D, result.ID);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	ResourceManager::createTexture(image);

	cachedTextures[texture] = result;

	return result;
}
ResourceManager::AtlasLoadingResult ResourceManager::loadAtlas(std::string &atlaspath)
{
	auto iter = cachedAtlases.find(atlaspath);

	if (iter != cachedAtlases.end())
	{
		return iter->second;
	}

	AtlasLoadingResult result;
	result.ID = 0;
	
	auto path = fs::path(resourceRoot / atlaspath);
	auto image = cv::imread(path.string(), cv::IMREAD_UNCHANGED);

	if (image.empty())
	{
		return result;
	}

	try
	{
		path = path.replace_extension(".json");
		boost::property_tree::ptree meta;
		boost::property_tree::json_parser::read_json(path.string(), meta);

		result.count = meta.get<size_t>("count");
		auto size = JSONVectorReader::readVector2f(meta.get_child("imagesSize"));
		size[0] /= image.cols;
		size[1] /= image.rows;
		result.atlasTextureSize = size;

		for (auto &record : meta.get_child("images"))
		{
			auto shift = JSONVectorReader::readVector2f(record.second.get_child("shift"));
			shift[0] /= image.cols;
			shift[1] /= image.rows;
			AtlasLoadingResult::AtlasTexture atlasTexture = { shift };

			result.atlasTextures.push_back(atlasTexture);
		}
	}
	catch (...)
	{
		result.ID = 0;
		return result;
	}

	result.atlasWidth = image.cols;
	result.atlasHeight = image.rows;

	glGenTextures(1, &result.ID);
	glBindTexture(GL_TEXTURE_2D, result.ID);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	ResourceManager::createTexture(image);

	cachedAtlases[atlaspath] = result;

	return result;
}

ResourceManager::AnimationLoadingResult ResourceManager::loadAnimation(std::string &animationFolder)
{

	auto iter = cachedAnimations.find(animationFolder);

	if (iter != cachedAnimations.end())
	{
		return iter->second;
	}

	AnimationLoadingResult result;

	bool animationSizeIsLoaded = false;

	std::vector<GLuint> framesIDs;

	int i = 0;

	std::string path = animationFolder + std::to_string(i) + ".png";
	GLuint texID = 0;
	auto image = cv::imread(fs::path(resourceRoot / path).string(), cv::IMREAD_UNCHANGED);

	while (!image.empty())
	{
		if (!animationSizeIsLoaded)
		{
			result.animationWidth = image.cols;
			result.animationHeight = image.rows;

			animationSizeIsLoaded = true;
		}

		glGenTextures(1, &texID);
		glBindTexture(GL_TEXTURE_2D, texID);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		ResourceManager::createTexture(image);

		result.IDs.push_back(texID);

		path = animationFolder + std::to_string(++i) + ".png";
		texID = 0;
		image = cv::imread(fs::path(resourceRoot / path).string(), cv::IMREAD_UNCHANGED);
	}

	result.animationLength = i;

	cachedAnimations[animationFolder] = result;

	return result;
}

GLuint ResourceManager::loadCubemap(std::array<std::string, 6> &cubemapTextures)
{
	auto iter = cachedCubemaps.find(cubemapTextures[0]);

	if (iter != cachedCubemaps.end())
	{
		return iter->second;
	}

	GLuint cubemapID;

	if (resourceRoot != "")
	{
		for (int i = 0; i < 6; ++i)
		{
			cubemapTextures[i] = fs::path(resourceRoot / cubemapTextures[i]).string();
		}
	}

	cubemapID = createCubemap(cubemapTextures);

	cachedCubemaps[cubemapTextures[0]] = cubemapID;

	return cubemapID;
}

void ResourceManager::removeTextures(std::vector<std::string> &textures)
{
	std::vector<GLuint> IDs(textures.size());

	for (auto &texture : textures)
	{
		auto iter = cachedTextures.find(texture);

		if (iter == cachedTextures.end())
		{
			continue;
		}

		IDs.push_back(iter->second.ID);
		cachedTextures.erase(iter);
	}

	glDeleteTextures(IDs.size(), IDs.data());
}

void ResourceManager::removeAnimation(std::string &animationPath)
{
	auto iter = cachedAnimations.find(animationPath);

	if (iter == cachedAnimations.end())
	{
		return;
	}

	glDeleteTextures(iter->second.IDs.size(), iter->second.IDs.data());

	cachedAnimations.erase(iter);
}

void ResourceManager::removeCubemap(std::array<std::string, 6> &cubemapTextures)
{
	auto iter = cachedCubemaps.find(cubemapTextures[0]);

	if (iter == cachedCubemaps.end())
	{
		return;
	}

	glDeleteTextures(1, &iter->second);

	cachedCubemaps.erase(iter);
}

void ResourceManager::clear()
{
	resourceRoot = "";

	std::vector<GLuint> IDs;

	for (auto &texture : cachedTextures)
	{
		IDs.push_back(texture.second.ID);
	}

	glDeleteTextures(IDs.size(), IDs.data());

	for (auto &animation : cachedAnimations)
	{
		glDeleteTextures(animation.second.IDs.size(), animation.second.IDs.data());
	}

	for (auto &cubemap : cachedCubemaps)
	{
		glDeleteTextures(1, &cubemap.second);
	}

	cachedTextures.clear();
	cachedAnimations.clear();
	cachedCubemaps.clear();
}