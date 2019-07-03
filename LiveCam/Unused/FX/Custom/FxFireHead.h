#include <fx\FxAnimatedStandardEmojiAncestor.h>
#include <fx\FxWidgetFacePaint.h>

class FxFireHead : public FX
{
public:

	FxWidget2DAtlas* fire;

	FxFireHead();
	~FxFireHead();

	void transform(FXModel& face, ExternalRenderParams &externalRenderParams) override;
};