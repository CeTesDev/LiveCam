#pragma once

#include <vector>

class ValueSmoother
{
public:

    static double SMOOTH_MODIFIER_Z;
    static double SMOOTH_MODIFIER_XY;

    ValueSmoother();

    void reset();

    double smooth(double newValue);

    bool inited = false;

    double SMOOTH_MODIFIER;

    double smoothValue = 0;

protected:
};
