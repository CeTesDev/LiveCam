#include "ValueSmoother.h"

#include <array>
#include <chrono>

using namespace std;

using namespace std::chrono;

double ValueSmoother::SMOOTH_MODIFIER_XY = 30;
double ValueSmoother::SMOOTH_MODIFIER_Z = 60;

ValueSmoother::ValueSmoother()
{
    SMOOTH_MODIFIER = 1.0 / SMOOTH_MODIFIER_XY / 2;
}

float snapCurve(float x)
{
    float y = 1.0 / (x + 1.0);
    y = (1.0 - y) * 2.0;
    if (y > 1.0)
    {
        return 1.0;
    }
    return y;
}

void ValueSmoother::reset()
{
    inited = false;
}

double ValueSmoother::smooth(double newValue)
{
    if (!inited)
    {
        inited = true;
        smoothValue = newValue;
        return smoothValue;
    }
    auto diff = abs(newValue - smoothValue);
    float snap = snapCurve(diff * SMOOTH_MODIFIER);

    smoothValue += (newValue - smoothValue) * snap;
    return smoothValue;
}
