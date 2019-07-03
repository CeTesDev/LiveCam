#include "Detector.h"
#include <math.h>
#include <Common/CommonClasses.h>
#include <Common/Resolutions.h>
#include <Tracking/TrackingTarget.h>
#include <Tracking/TrackingStructs.h>
#include <Tracking/DetectorThreadPool.h>

#include <Tracking/FaceScanner.h>
#include <Tracking/FaceKeypointScanner.h>

#ifdef LC_INCLUDE_SURFACE_TRACKING
#include <Tracking/SurfaceScanner.h>
#endif

#ifdef LC_INCLUDE_HAND_TRACKING
#include <Tracking/HandScanner.h>
#endif

extern FX *currentDetectEffect;

DetectorThreadPool threadPool(MAX_TARGETS_COUNT);

float calcSquaredDistanceBetweenFaces(const cv::Rect& face1, const cv::Rect& face2)
{
	cv::Point2f oldCenter(face1.x + face1.width*0.5f, face1.y + face1.height*0.5f);
	cv::Point2f newCenter(face2.x + face2.width*0.5f, face2.y + face2.height*0.5f);

	float distanceX = newCenter.x - oldCenter.x;
	float distanceY = newCenter.y - oldCenter.y;
	float squareDistance = distanceX * distanceX + distanceY * distanceY;

	return squareDistance;
}

Detector::Detector()
{
	isRestricted = false;
	activeTrackingType = TRACKER_TYPE::FACE;
	restrictedLimit = MAX_TARGETS_COUNT;
	unRestrictedLimit = MAX_TARGETS_COUNT;
	targetLimit = MAX_TARGETS_COUNT;
}

Detector::~Detector()
{
	stop();
}

void Detector::init(const std::string& targetTracker, const std::string& detailHandler, int lostTargetSleepMs, int lostTargetDelayMs, double scaleFactor)
{
	this->lostTargetDelayMs = lostTargetDelayMs < lostTargetSleepMs ? lostTargetDelayMs : lostTargetSleepMs;
	this->lostTargetSleepMs = lostTargetSleepMs;
	this->scaleModifier = scaleFactor;

	std::string trackerLib = targetTracker;
	boost::to_upper(trackerLib);
	std::string::iterator end_pos = std::remove(trackerLib.begin(), trackerLib.end(), ' ');
	trackerLib.erase(end_pos, trackerLib.end());

	if (TRACKER_LIBMAP.find(trackerLib) != TRACKER_LIBMAP.end())
	{
		TRACKER_LIB libKey = TRACKER_LIBMAP.at(trackerLib);
		scanners.insert({ TRACKER_TYPE::FACE, std::make_shared<FaceScanner>(libKey) });
		scanner = scanners.at(TRACKER_TYPE::FACE);
		scanner->scale(scaleFactor);
	}
	else
	{
		throw std::exception("Invalid face detector type.");
	}

	if (detailHandler == "Caffe" || detailHandler == "Caffe3D")
	{
		for (auto &ptr : keypointScanners)
		{
			ptr = std::make_shared<FaceKeypointScanner>((detailHandler == "Caffe3D"));
		}
	}
	else
	{
		throw std::exception("Invalid landmark detector type.");
	}

	cv::setNumThreads(4);

#if defined(LC_INCLUDE_HAND_TRACKING) || defined(LC_INCLUDE_SURFACE_TRACKING)
	std::function<void(void)> pfn = std::bind(&Detector::loadExtraTrackers, this);
	threadPool.enqueue(pfn);
#endif

	scanner->start();
}

void Detector::start()
{
	if (started)
	{
		return;
	}
	syncStop = false;
	for (size_t i = 0; i < MAX_TARGETS_COUNT; i++)
	{
		_syncRects[i].visible = false;
		_syncDetails[i].visible = false;
	}
	syncLastFrame = cv::Mat();
	started = true;
	if (scanner != nullptr)
	{
		scanner->start();
	}
}

void Detector::stop()
{
	if (!started)
	{
		return;
	}
	syncStop = true;
	started = false;
	if (scanner != nullptr)
	{
		scanner->stop();
	}
}

void Detector::restart()
{
	if (scanner != nullptr)
	{
		std::lock_guard<std::mutex> lock(scannerMutex);
		scanner->stop();
		reset();
		scanner->start();
	}
}

void Detector::reset()
{
	std::lock_guard<std::mutex> lock(targetMutex);
	for (int i = 0; i < MAX_TARGETS_COUNT; ++i)
	{
		_syncDetails[i].visible = false;
		_syncRects[i].visible = false;
		_tmpRects[i].visible = false;
	}
}

void Detector::loadExtraTrackers()
{
#ifdef LC_INCLUDE_HAND_TRACKING
	scanners.insert({ TRACKER_TYPE::HANDS, std::make_shared<HandScanner>() });
#endif
#ifdef LC_INCLUDE_SURFACE_TRACKING
	scanners.insert({ TRACKER_TYPE::SURFACE, std::make_shared<SurfaceScanner>(true) });
#endif
}

void Detector::setTrackingType(std::string typeString)
{
	requestedTrackingType = typeString;
	boost::to_upper(typeString);
	std::string::iterator end_pos = std::remove(typeString.begin(), typeString.end(), ' ');
	typeString.erase(end_pos, typeString.end());
	TRACKER_TYPE key = TRACKER_TYPEMAP.at(typeString);
	if (scanners.find(key) != scanners.end())
	{
		if (activeTrackingType != key || scanner == nullptr)
		{
			{
				std::unique_lock<std::mutex> lock(scannerMutex);
				activeTrackingType = key;
				scanner = scanners.at(key);
				minimalDetail = key != TRACKER_TYPE::FACE;
			}
			restart();
		}
	}
}

void Detector::setTargetLimit(int value)
{
	std::unique_lock<std::mutex> lock(targetLimitMutex);

	if (value < 1)
	{
		value = 1;
	}

	if (value > MAX_TARGETS_COUNT)
	{
		value = MAX_TARGETS_COUNT;
	}

	if (scanner != nullptr)
	{
		if (value > scanner->capacity)
		{
			value = scanner->capacity;
		}
	}

	unRestrictedLimit = value;

	if (!isRestricted || value < restrictedLimit)
	{
		targetLimit = value;
		int threadCount = (value > 5) ? 4 : 1;
		cv::setNumThreads(threadCount);
	}
}

void Detector::setTargetRestriction(int value)
{
	std::unique_lock<std::mutex> lock(targetLimitMutex);

	if (value < 1)
	{
		isRestricted = false;
		restrictedLimit = unRestrictedLimit;
	}
	else
	{
		if (value > MAX_TARGETS_COUNT)
		{
			value = MAX_TARGETS_COUNT;
		}
		isRestricted = true;
		restrictedLimit = value;
	}

	targetLimit = restrictedLimit;

	int threadCount = (targetLimit > 5) ? 4 : 1;
	cv::setNumThreads(threadCount);
}

std::vector<TrackingTarget> Detector::getData(cv::Mat frame)
{
	if (scanner == nullptr)
	{
		setTrackingType(requestedTrackingType);
	}
	else if (!frame.empty())
	{
		{
			std::lock_guard<std::mutex> scannerLock(scannerMutex);
			std::lock_guard<std::mutex> frameLock(frameMutex);
			currentRows = frame.rows;
			currentCols = frame.cols;
			scanImage = scanner->prepScanFrame(frame);
			keypointImage = minimalDetail ? keypointScanners[0]->preProcess(frame) : scanner->prepKeypointsFrame(frame);
		}

		std::function<void(void)> pfn = std::bind(&Detector::scanTargets, this);
		threadPool.enqueue(pfn);
	}

	return targetData;
}

void Detector::setSnapshot(cv::Mat frame, int x, int y)
{
	if (scanner == nullptr)
	{
		setTrackingType(requestedTrackingType);
	}
	scanner->setRefFrame(frame, x, y);
}

void Detector::scanTargets()
{
	static std::array<TrackingTarget, MAX_TARGETS_COUNT> previous;
	std::vector<TrackingTarget> resultArr;

	auto details = getDetailData();

	for (size_t i = 0; i < details.size(); i++)
	{
		auto targetDetails = details[i];

		if (targetDetails.visible)
		{
			TrackingTarget target;

			float imgWidth = currentCols;
			float imgHeight = currentRows;
			float widthDiff = 1.0f;
			float heightDiff = 1.0f;

			if (imgWidth != Resolutions::OUTPUT_WIDTH) {
				widthDiff = Resolutions::OUTPUT_WIDTH / imgWidth;
			}

			if (imgHeight != Resolutions::OUTPUT_HEIGHT) {
				heightDiff = Resolutions::OUTPUT_HEIGHT / imgHeight;
			}

			target.rect = targetDetails.rect;
			target.lastKnownRect = targetDetails.lastKnownRect;

			target.pointId = i;
			target.pointTotal = TARGET_DETAIL_MODIFIER;
			target.pts.fill(0);

			target.frameWidth = currentCols;
			target.frameHeight = currentRows;

			target.inited = true;

			targetDetails.getProjection(i);

			if (targetDetails.trackerType == TRACKER_TYPE::FACE)
			{
				if (abs(targetDetails.eulerAngles[2]) < 70)
				{
					target.rotation <<
						targetDetails.rotationMatrix.at<double>(0, 0),
						targetDetails.rotationMatrix.at<double>(0, 1),
						targetDetails.rotationMatrix.at<double>(0, 2),
						targetDetails.rotationMatrix.at<double>(1, 0),
						targetDetails.rotationMatrix.at<double>(1, 1),
						targetDetails.rotationMatrix.at<double>(1, 2),
						targetDetails.rotationMatrix.at<double>(2, 0),
						targetDetails.rotationMatrix.at<double>(2, 1),
						targetDetails.rotationMatrix.at<double>(2, 2);

					for (size_t j = 0; j < TOTAL_DETAIL_POINTS; ++j)
					{
						target.pts[2 * j] = targetDetails.data[j].x;
						target.pts[2 * j + 1] = targetDetails.data[j].y;
					}

					target.widthRaw = PointCalcUtil::distanceBetween(targetDetails.data[0], targetDetails.data[16]);
					float width = target.widthRaw / imgWidth;

					target.width = width;

					target.xCenterRaw = target.pts[58];
					target.yCenterRaw = target.pts[59];

					target.xCenter = (float)-(1 - (target.xCenterRaw / 2) / (imgWidth / 2));
					target.yCenter = (float)(1 - (target.yCenterRaw / 2) / (imgHeight / 2));

					float roll = -targetDetails.eulerAngles(2);
					float pitch = targetDetails.eulerAngles(0);
					float yaw = -targetDetails.eulerAngles(1);

					target.pitch = pitch;
					target.roll = roll;
					target.yaw = yaw;

					previous[i] = target;
					resultArr.push_back(target);
				}
				else
				{
					resultArr.push_back(previous[i]);
				}
			}
			else
			{
				if (targetDetails.trackerType == TRACKER_TYPE::HANDS)
				{
					for (size_t j = 0; j < TOTAL_DETAIL_POINTS; ++j)
					{
						target.pts[2 * j] = targetDetails.data[j].x;
						target.pts[2 * j + 1] = targetDetails.data[j].y;
					}
					target.widthRaw = targetDetails.rect.width * 6;
					//float height = targetDetails.rect.height * 4;
					target.xCenterRaw = targetDetails.data[2 * i + 1].x - (targetDetails.rect.width / 2);
					target.yCenterRaw = targetDetails.data[2 * i + 1].y - targetDetails.rect.width;
				}
				else
				{
					target.xCenterRaw = targetDetails.rect.x;
					target.yCenterRaw = targetDetails.rect.y;
					target.widthRaw = targetDetails.rect.width;
				}

				float width = target.widthRaw / imgWidth;

				target.width = width;

				target.xCenter = (float)-(1 - (target.xCenterRaw / 2) / (imgWidth / 2));
				target.yCenter = (float)(1 - (target.yCenterRaw / 2) / (imgHeight / 2));

				target.pitch = 0;
				target.roll = 0;
				target.yaw = 0;

				resultArr.push_back(target);
			}
		}
	}
	{
		std::lock_guard<std::mutex> lock2(targetMutex);
		this->targetData = resultArr;
	}
}

std::vector<TargetDetailStruct> Detector::getDetailData()
{
	std::vector<TargetDetailStruct> result;

	if (this->threadAvailable)
	{
		this->threadAvailable = false;

		if (!scanImage.empty())
		{
			int localFrameIndex = 0;
			std::unique_lock<std::mutex> lock(frameMutex);
			localFrameIndex = syncLastFrameIndex;
			std::vector<cv::Rect> detected = scanner->scan(scanImage);
			std::vector<cv::Mat> frames;
			
			if (!minimalDetail)
			{
				for (size_t i = 0; i < detected.size(); )
				{
					if (detected[i].x < 0 || detected[i].y < 0 || (detected[i].x + detected[i].width >= scanImage.cols) || (detected[i].y + detected[i].height >= scanImage.rows))
					{
						detected.erase(detected.begin() + i);
					}
					else
					{
						i++;
					}
				}
				//keypointImage = keypointScanners[0]->preProcess(scanImage);
			}
			for (size_t i = 0; i < detected.size(); i++)
			{
				frames.push_back(keypointImage);
			}

			std::array<bool, MAX_TARGETS_COUNT> previousVisibility;
			for (int i = 0; i < MAX_TARGETS_COUNT; ++i)
			{
				previousVisibility[i] = _tmpRects[i].visible;
			}

			if (!minimalDetail)
			{
				sort(detected, frames, _tmpRects);

				for (int i = 0; i < MAX_TARGETS_COUNT; ++i)
				{
					if (!previousVisibility[i])
					{
						_syncDetails[i].visible = false;
					}
				}

				for (size_t i = 0; i < _tmpRects.size(); i++)
				{
					if (_tmpRects[i].visible)
					{
						_tmpRects[i].frameIndex = localFrameIndex;
					}
				}
			}
			else
			{
				for (size_t i = 0; i < _tmpRects.size(); i++)
				{
					if (i < detected.size() && detected[i].width > 0)
					{
						_tmpRects[i].visible = true;
						_tmpRects[i].rect = detected[i];
						_tmpRects[i].frame = frames[i];
						_tmpRects[i].frameIndex = localFrameIndex;
					}
					else
					{
						_tmpRects[i].visible = false;
					}
				}
			}
		}
		else
		{
			for (int i = 0; i < MAX_TARGETS_COUNT; ++i)
			{
				_syncDetails[i].visible = false;
			}
		}
		{
			_syncRects = _tmpRects;
		}

		TargetHoldingStruct sourceTarget;

		for (size_t i = 0; i < _syncRects.size(); i++) {

			TargetDetailStruct targetDetails;
			if (_syncRects[i].visible)
			{
				sourceTarget = _syncRects[i];
				if (activeTrackingType == TRACKER_TYPE::FACE)
				{
					keypointScanners[0]->scan(sourceTarget, targetDetails.data);
				}
				else
				{
					scanner->keypoints(sourceTarget, targetDetails.data);
					targetDetails.frame = sourceTarget.frame;
					targetDetails.lastKnownFrame = sourceTarget.lastKnownFrame;
				}
				targetDetails.minimalDetail = minimalDetail;
				targetDetails.trackerType = activeTrackingType;
				targetDetails.rect = sourceTarget.rect;
				targetDetails.lastKnownRect = sourceTarget.lastKnownRect;
				targetDetails.frameIndex = sourceTarget.frameIndex;
				targetDetails.visible = sourceTarget.visible;
				targetDetails.pointId = sourceTarget.pointId;
			}
			else
			{
				targetDetails.visible = false;
			}
			_syncDetails[i] = targetDetails;
		}

		auto endTime = std::chrono::system_clock::now();

		result.reserve(MAX_TARGETS_COUNT);

		{
			std::unique_lock<std::mutex> lock(frameMutex);
			syncLastFrame = keypointImage;
			syncLastFrameIndex = (syncLastFrameIndex + 1) % 1000;
		}
		{
			std::unique_lock<std::mutex> lock1(landmarkMutex);
			std::unique_lock<std::mutex> lock2(targetMutex);

			for (int i = 0; i < MAX_TARGETS_COUNT; ++i)
			{
				result.push_back(_syncDetails[i]);

				result.back().visible = _syncRects[i].visible & _syncDetails[i].visible;
			}
		}

		this->threadAvailable = true;
	}

	return result;
}

void Detector::getTargetData(cv::Mat frame)
{
	
}

void Detector::sort(std::vector<cv::Rect>& targets, std::vector<cv::Mat>& frames, std::array<TargetHoldingStruct, MAX_TARGETS_COUNT>& targetRegistry)
{
	for (size_t i = 0; i < MAX_TARGETS_COUNT; i++)
	{
		targetsFound[i].clear();
	}

	if (targets.size() == 0)
	{
		for (size_t j = 0; j < MAX_TARGETS_COUNT; j++)
		{
			targetRegistry[j].visible = false;
		}
		return;
	}

	for (int i = 0; i < (int)targets.size() - 1; i++)
	{
		for (int j = i + 1; j < targets.size(); ++j)
		{
			float distance = calcSquaredDistanceBetweenFaces(targets[j], targets[i]);

			bool Ibigger = targets[i].area() > targets[j].area();

			float biggerDiagonal = Ibigger
				?
				sqrtf(
					targets[i].width * targets[i].width +
					targets[i].height * targets[i].height)
				:
				sqrtf(
					targets[j].width * targets[j].width +
					targets[j].height * targets[j].height)
				;

			float distanceRatio = 2 * sqrtf(distance) / biggerDiagonal;

			if (distanceRatio < 0.5f)
			{
				if (Ibigger)
				{
					targets.erase(targets.begin() + j--);
					continue;
				}
				else
				{
					targets.erase(targets.begin() + i--);
					break;
				}
			}
		}
	}

	if (targets.size() == 0)
	{
		return;
	}

	auto now = std::chrono::system_clock::now();

	std::array<bool, MAX_TARGETS_COUNT> bucketAvailable;
	std::array<bool, MAX_TARGETS_COUNT> bucketFree;
	bucketFree.fill(true);

	for (int i = 0; i < MAX_TARGETS_COUNT; i++)
	{
		bucketAvailable[i] = now - targetRegistry[i].lastKnownTime < std::chrono::milliseconds(lostTargetSleepMs);
	}

	for (int i = 0; i < MAX_TARGETS_COUNT - 1; i++)
	{
		if (!bucketAvailable[i]) continue;

		for (int j = i + 1; j < MAX_TARGETS_COUNT; ++j)
		{
			if (!bucketAvailable[j]) continue;

			float distance = calcSquaredDistanceBetweenFaces(targetRegistry[j].rect, targetRegistry[i].rect);

			bool Ibigger = targetRegistry[i].rect.area() > targetRegistry[j].rect.area();

			float biggerDiagonal = Ibigger
				?
				sqrtf(
					targetRegistry[i].rect.width * targetRegistry[i].rect.width +
					targetRegistry[i].rect.height * targetRegistry[i].rect.height)
				:
				sqrtf(
					targetRegistry[j].rect.width * targetRegistry[j].rect.width +
					targetRegistry[j].rect.height * targetRegistry[j].rect.height)
				;

			float distanceRatio = 2 * sqrtf(distance) / biggerDiagonal;

			if (distanceRatio < 0.5f)
			{
				auto timeDiffI = std::chrono::duration_cast<std::chrono::milliseconds>(now - targetRegistry[i].lastKnownTime);
				auto timeDiffJ = std::chrono::duration_cast<std::chrono::milliseconds>(now - targetRegistry[j].lastKnownTime);

				if (timeDiffI != timeDiffJ)
				{
					if (timeDiffI < timeDiffJ)
					{
						bucketAvailable[j] = false;
						bucketFree[j] = false;
						continue;
					}
					else
					{
						bucketAvailable[i] = false;
						bucketFree[i] = false;
						continue;
					}
				}
				else
				{
					if (Ibigger)
					{
						bucketAvailable[j] = false;
						bucketFree[j] = false;
						continue;
					}
					else
					{
						bucketAvailable[i] = false;
						bucketFree[i] = false;
						continue;
					}
				}
			}
		}
	}

	std::vector<int> buckets(targets.size(), -1);
	std::vector<float> distances(targets.size());

	bool sorting;

	do
	{
		for (size_t i = 0; i < targets.size(); ++i)
		{
			if (buckets[i] != -1)
			{
				continue;
			}

			float lastDistance = std::numeric_limits<float>::max();
			int lastIndex = -1;

			float newDiagonal = sqrtf(
				targets[i].width * targets[i].width +
				targets[i].height * targets[i].height);

			for (size_t j = 0; j < MAX_TARGETS_COUNT; j++)
			{
				if (!bucketAvailable[j])
				{
					continue;
				}

				float newDistance = calcSquaredDistanceBetweenFaces(targetRegistry[j].rect, targets[i]);

				float oldDiagonal = sqrtf(
					targetRegistry[j].rect.width * targetRegistry[j].rect.width +
					targetRegistry[j].rect.height * targetRegistry[j].rect.height);

				float diagonalRatio = newDiagonal / oldDiagonal;
				float distanceRatio = 2 * sqrtf(newDistance) / oldDiagonal;

				if (newDistance < lastDistance &&
					diagonalRatio > 0.77f && diagonalRatio < 1.3f
					&& distanceRatio < 2.f
					)
				{
					lastDistance = newDistance;
					lastIndex = j;
				}
			}

			if (lastIndex != -1)
			{
				buckets[i] = lastIndex;
				distances[i] = lastDistance;
			}
		}

		sorting = false;

		for (size_t i = 0; i < targets.size() - 1; ++i)
		{
			if (buckets[i] == -1)
			{
				continue;
			}

			for (size_t p = i + 1; p < targets.size(); ++p)
			{
				if (buckets[p] == -1)
				{
					continue;
				}

				if (buckets[i] == buckets[p])
				{
					sorting = true;

					if (distances[i] < distances[p])
					{
						buckets[p] = -1;
					}
					else
					{
						buckets[i] = -1;
					}
				}
			}
		}

		for (size_t i = 0; i < targets.size(); ++i)
		{
			int bucketIndex = buckets[i];

			if (bucketIndex != -1)
			{
				bucketAvailable[bucketIndex] = false;
				bucketFree[bucketIndex] = false;
			}
		}

	} while (sorting);

	for (size_t i = 0; i < targets.size(); )
	{
		if (buckets[i] != -1)
		{
			targetsFound[buckets[i]].push_back(std::make_pair(targets[i], frames[i]));
			targets.erase(targets.begin() + i);
			frames.erase(frames.begin() + i);
			buckets.erase(buckets.begin() + i);
		}
		else
		{
			++i;
		}
	}

	std::unique_lock<std::mutex> lock(targetLimitMutex);

	int activeFaceCounter = 0;

	for (size_t j = 0; j < MAX_TARGETS_COUNT; j++)
	{
		if (now - targetRegistry[j].lastKnownTime >= std::chrono::milliseconds(lostTargetSleepMs) || activeFaceCounter == targetLimit)
		{
			targetRegistry[j].visible = false;
		}
		else
		{
			targetRegistry[j].visible = targetsFound[j].size() != 0 || now - targetRegistry[j].lastKnownTime < std::chrono::milliseconds(lostTargetDelayMs);

			++activeFaceCounter;

			if (targetsFound[j].size() > 0)
			{
				int lastIndex = 0;

				targetRegistry[j].rect = targetsFound[j][lastIndex].first;
				targetRegistry[j].frame = targetsFound[j][lastIndex].second;

				for (size_t k = 0; k < targetsFound[j].size(); k++)
				{
					if (k != lastIndex)
					{
						targets.push_back(targetsFound[j][k].first);
						frames.push_back(targetsFound[j][k].second);
					}
				}
			}
		}
	}

	for (size_t j = 0; j < MAX_TARGETS_COUNT; j++)
	{
		if (targetRegistry[j].visible && targetsFound[j].size() > 0)
		{
			targetRegistry[j].lastKnownTime = now;
		}
	}

	if (activeFaceCounter == targetLimit)
	{
		return;
	}

	for (int i = 0; i < MAX_TARGETS_COUNT; i++)
	{
		if (!targetRegistry[i].visible) continue;

		for (int j = 0; j < targets.size(); ++j)
		{
			float distance = calcSquaredDistanceBetweenFaces(targets[j], targetRegistry[i].rect);

			float diagonal = sqrtf(
				targetRegistry[i].rect.width * targetRegistry[i].rect.width +
				targetRegistry[i].rect.height * targetRegistry[i].rect.height)
				;

			float distanceRatio = 2 * sqrtf(distance) / diagonal;

			if (distanceRatio < 0.5f)
			{
				targets.erase(targets.begin() + j--);
			}
		}
	}

	if (targets.empty())
	{
		return;
	}

	if (targets.size() > targetLimit - activeFaceCounter)
	{
		targets.resize(targetLimit - activeFaceCounter);
	}

	buckets.resize(targets.size(), -1);
	distances.resize(targets.size());

	do
	{
		for (size_t i = 0; i < targets.size(); ++i)
		{
			if (buckets[i] != -1)
			{
				continue;
			}

			float lastDistance = std::numeric_limits<float>::max();
			int lastIndex = -1;

			for (size_t j = 0; j < MAX_TARGETS_COUNT; j++)
			{
				if (!bucketFree[j])
				{
					continue;
				}

				float newDistance = calcSquaredDistanceBetweenFaces(targetRegistry[j].rect, targets[i]);

				if (newDistance < lastDistance)
				{
					lastDistance = newDistance;
					lastIndex = j;
				}
			}

			if (lastIndex != -1)
			{
				buckets[i] = lastIndex;
				distances[i] = lastDistance;
			}
		}

		sorting = false;

		for (size_t i = 0; i < targets.size() - 1; ++i)
		{
			if (buckets[i] == -1)
			{
				continue;
			}

			for (size_t p = i + 1; p < targets.size(); ++p)
			{
				if (buckets[p] == -1)
				{
					continue;
				}

				if (buckets[i] == buckets[p])
				{
					sorting = true;

					if (distances[i] < distances[p])
					{
						buckets[p] = -1;
					}
					else
					{
						buckets[i] = -1;
					}
				}
			}
		}

		for (size_t i = 0; i < targets.size(); ++i)
		{
			int bucketIndex = buckets[i];

			if (bucketIndex != -1)
			{
				bucketFree[bucketIndex] = false;
			}
		}

	} while (sorting);

	for (size_t i = 0; i < targets.size(); ++i)
	{
		int bucketIndex = buckets[i];
		targetRegistry[bucketIndex].visible = true;
		targetRegistry[bucketIndex].rect = targets[i];
		targetRegistry[bucketIndex].lastKnownFrame = targetRegistry[bucketIndex].frame;
		targetRegistry[bucketIndex].frame = frames[i];
		targetRegistry[bucketIndex].lastKnownTime = now;
	}
}
