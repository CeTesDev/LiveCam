#pragma once

#include <map>
#include <string>
#include <memory>

#include <Particles/Particles.h>

class ParticleManager
{
public:

	ParticleManager();
	~ParticleManager();

	std::shared_ptr<ParticleEffect> addEffect(std::string& path);
};
