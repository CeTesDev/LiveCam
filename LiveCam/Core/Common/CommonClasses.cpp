#include <Common/CommonClasses.h>

std::shared_ptr<cwc::glShader> ShaderManagerWrapper::LoadFromFile(const char* vertexFile, const char* fragmentFile)
{

	std::string key = std::string(vertexFile) + fragmentFile;

	if (cachedShaders.find(key) != cachedShaders.end())
	{
		return cachedShaders[key];
	}

	std::shared_ptr<cwc::glShader> result;

	if (!shaderManager)
	{
		shaderManager = std::make_shared<cwc::glShaderManager>();
	}

	cwc::glShader* shaderPtr = shaderManager->loadfromFile(vertexFile, fragmentFile);

	if (shaderPtr)
	{
		result = std::shared_ptr<cwc::glShader>(shaderPtr);

		cachedShaders[key] = result;
	}
	else
	{
		qDebug() << "Error Loading, compiling or linking shader\n";
	}

	return result;
}

void ShaderManagerWrapper::Clear()
{
	cachedShaders.clear();
	shaderManager.reset();
}