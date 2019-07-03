#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

#include <Common/CommonClasses.h>
#include <Tracking/TrackingTarget.h>
#include <Graphics/GraphicsModel.h>
#include <FX.h>

class FxWidgetSprite
{
public:
	FxWidgetSprite();

	virtual void load();
	virtual void draw(Eigen::Matrix4f &externalModelMatrix, Eigen::Matrix4f &externalProjectionMatrix);
	virtual void unload();

	virtual void animate();

	GraphicsData object;

	std::shared_ptr<cwc::glShader> shader;

	Eigen::Matrix4f modelMatrix;

	std::string animationPath;
	Eigen::Vector2f animationSize;

	Eigen::Vector3f animationOffset;
	Eigen::Vector2f animationScale;

	size_t animationLength;

	std::vector<GLuint>animationFramesIDs;
	size_t animationIndex;
};