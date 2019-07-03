#include "LiveCamWindow.h"

#include <Common/Resolutions.h>

#include "../main.h"
#include <qscrollbar.h>

extern Detector detector;

Q_DECLARE_METATYPE(QCameraInfo)

const QString FilterButton::uncheckedStylesheet = "#filterButton { border-bottom-left-radius: 15px; border-bottom-right-radius: 15px; }"
"#filterButton:hover { background-color: rgb(35, 162, 255); }";

const QString FilterButton::checkedStylesheet = "#filterButton { border-bottom-left-radius: 15px; border-bottom-right-radius: 15px;"
"background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 rgb(27, 47, 100), stop:1 rgb(64, 84, 188)); }"
"#filterButton:hover { background-color: rgb(35, 162, 255); }";

FilterButton::FilterButton(const QString buttonName, const QString iconPath, LiveCamWindow* liveCamWindow) : QWidget(liveCamWindow)
{
	this->liveCamWindow = liveCamWindow;
	this->setObjectName("filterButton");

	this->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	this->setStyleSheet(FilterButton::uncheckedStylesheet);

	auto verticalLayout = new QVBoxLayout(this);
	verticalLayout->setContentsMargins(30, 10, 30, 15);
	verticalLayout->setSpacing(10);

	iconLabel = new QLabel(this);
	iconLabel->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	iconLabel->setMinimumSize(60, 60);

	nameLabel = new QLabel(this);
	nameLabel->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	nameLabel->setText(buttonName);
	nameLabel->setAlignment(Qt::AlignmentFlag::AlignVCenter | Qt::AlignmentFlag::AlignHCenter);

	verticalLayout->addWidget(iconLabel, 0, Qt::AlignmentFlag::AlignHCenter);
	verticalLayout->addWidget(nameLabel, 0, Qt::AlignmentFlag::AlignHCenter);
	this->setLayout(verticalLayout);
}

void FilterButton::paintEvent(QPaintEvent *)
{
	QStyleOption opt;
	opt.init(this);
	QPainter painter(this);
	style()->drawPrimitive(QStyle::PE_Widget, &opt, &painter, this);
}

void FilterButton::mousePressEvent(QMouseEvent *event)
{
	if (this == liveCamWindow->activeFilterButton)
	{
		return;
	}

	if (liveCamWindow->activeFilterButton)
	{
		liveCamWindow->activeFilterButton->setChecked(false);
	}

	setChecked(true);

	liveCamWindow->on_filterButton_clicked(this);
}

void FilterButton::setChecked(bool value)
{
	setStyleSheet(value ? checkedStylesheet : uncheckedStylesheet);
}

void FilterButton::setIcon(QIcon icon)
{
	iconLabel->setPixmap(icon.pixmap(QSize(60, 60)));
}



const int LabelShapedButton::verticalPadding = 5;
const int LabelShapedButton::horizontalPadding = 30;
const QFont LabelShapedButton::font("Segoe UI", 12);
const std::string deckLinkString("Decklink");
const std::string blackMagicString("Blackmagic");

LabelShapedButton::LabelShapedButton(QString text, bool triangleAtRightSide, QColor textColor, QColor topColor, QColor bottomColor, QWidget* parent)
	: textMeasurer(font)
	, QPushButton(parent)
{
	this->textColor = textColor;

	setFont(font);
	setText(text);

	setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	setMinimumSize(150, textMeasurer.height() + 2 * verticalPadding);

	if (triangleAtRightSide)
	{
		QVector<QPoint> points = QVector<QPoint>(5);
		points[0] = QPoint(0, 0);
		points[1] = QPoint(0, height());
		points[2] = QPoint(width() - height() / 2.f, height());
		points[3] = QPoint(width(), height() - height() / 2.f);
		points[4] = QPoint(width() - height() / 2.f, 0);

		path.moveTo(points[0]);
		path.lineTo(points[1]);
		path.lineTo(points[2]);
		path.lineTo(points[3]);
		path.lineTo(points[4]);
		path.lineTo(points[0]);
	}
	else
	{
		QVector<QPoint> points = QVector<QPoint>(4);
		points[0] = QPoint(0, 0);
		points[1] = QPoint(0, height());
		points[2] = QPoint(width(), height());
		points[3] = QPoint(width(), 0);

		path.moveTo(points[0]);
		path.lineTo(points[1]);
		path.lineTo(points[2]);
		path.lineTo(points[3]);
		path.lineTo(points[0]);
	}

	int CIRCLE_RADIUS = height() / 6.f;
	int BORDER_RADIUS = 6;
	region = { rect(), QRegion::Rectangle };

	QRegion cornerRect;
	QRegion cornerEllipse;

	cornerRect = { 0, 0, BORDER_RADIUS, BORDER_RADIUS, QRegion::Rectangle };
	cornerEllipse = { 0, 0, BORDER_RADIUS * 2 + 1, BORDER_RADIUS * 2 + 1, QRegion::Ellipse };
	region = region.subtracted(cornerRect.subtracted(cornerEllipse));

	cornerRect = { 0, height() - BORDER_RADIUS, BORDER_RADIUS, BORDER_RADIUS, QRegion::Rectangle };
	cornerEllipse = { 0, height() - 1 - BORDER_RADIUS * 2 - 1, BORDER_RADIUS * 2 + 1, BORDER_RADIUS * 2 + 1, QRegion::Ellipse };
	region = region.subtracted(cornerRect.subtracted(cornerEllipse));

	cornerRect = { width() - BORDER_RADIUS, 0, BORDER_RADIUS, BORDER_RADIUS, QRegion::Rectangle };
	cornerEllipse = { width() - 1 - BORDER_RADIUS * 2 - 1, 0, BORDER_RADIUS * 2 + 1, BORDER_RADIUS * 2 + 1, QRegion::Ellipse };
	region = region.subtracted(cornerRect.subtracted(cornerEllipse));

	cornerRect = { width() - BORDER_RADIUS, height() - BORDER_RADIUS, BORDER_RADIUS, BORDER_RADIUS, QRegion::Rectangle };
	cornerEllipse = { width() - 1 - BORDER_RADIUS * 2 - 1, height() - 1 - BORDER_RADIUS * 2 - 1, BORDER_RADIUS * 2 + 1, BORDER_RADIUS * 2 + 1, QRegion::Ellipse };
	region = region.subtracted(cornerRect.subtracted(cornerEllipse));

	region = region.subtracted({ height() / 2 - CIRCLE_RADIUS, height() / 2 - CIRCLE_RADIUS, 2 * CIRCLE_RADIUS + 1, 2 * CIRCLE_RADIUS + 1, QRegion::Ellipse });

	gradient = { rect().topLeft(), rect().bottomRight() };
	gradient.setColorAt(0, topColor);
	gradient.setColorAt(1, bottomColor);
}

void LabelShapedButton::paintEvent(QPaintEvent *event)
{
	QPainter painter(this);

	painter.setRenderHint(QPainter::Antialiasing, true);

	painter.setClipRegion(region);

	painter.fillPath(path, gradient);

	painter.setFont(font);
	painter.setPen(textColor);
	painter.drawText(QPointF(height(), height() / 2.f + textMeasurer.height() / 4.f), text());
	qDebug() << textMeasurer.height();
	qDebug() << textMeasurer.ascent();
	qDebug() << textMeasurer.descent();

	//QPushButton::paintEvent(event);
}



void LiveCamWindow::addFilterButton(FilterButton* filterButton)
{
	ui.horizontalLayout_filterList->addWidget(filterButton);
}

QPushButton* LiveCamWindow::addModuleButton()
{
	QPushButton* moduleButton = new QPushButton();

	moduleButton->setVisible(false);
	moduleButton->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	moduleButton->setMinimumSize(80, 80);
	moduleButton->setIconSize({ 60, 60 });
	moduleButton->setMouseTracking(true);
	moduleButton->setStyleSheet("QWidget { border-radius: 15px; padding: 10px; image-position: top; }"
		"QWidget:hover { background-color: rgb(35, 162, 255); }");
	moduleButton->installEventFilter(this);

	ui.horizontalLayout_moduleList->addWidget(moduleButton);

	return moduleButton;
}

bool LiveCamWindow::event(QEvent * e) // overloading event(QEvent*) method of QMainWindow
{
	switch (e->type())
	{
	case QEvent::WindowActivate:
		//if (!ui.btn_startEffectLeft_Container->isEnabled())
		//{
		//	editorWindow->raise();
		//}
		break;
	};
	return QMainWindow::event(e);
}

LiveCamWindow::LiveCamWindow(QWidget *parent) : QMainWindow(parent)
{
	outputWindow = new OutputWindow(this);
	outputWindow->showMaximized();

	editorWindow = new EditorWindow(this);

	connect(editorWindow, SIGNAL(editor_window_closed()), this, SLOT(on_editor_window_closed()));
	connect(editorWindow->ui.filterIcon_button, SIGNAL(clicked()), this, SLOT(on_editor_filterIcon_changed()));
	connect(editorWindow->ui.filterIcon_delete, SIGNAL(clicked()), this, SLOT(on_editor_filterIcon_changed()));

	ui.setupUi(this);
	/*
	showFiltersButton = new LabelShapedButton("Show Filters", true, Qt::GlobalColor::white, QColor(52, 180, 243), QColor(19, 69, 218), this);
	startOutputButton = new LabelShapedButton("Start Output", false, Qt::GlobalColor::black, QColor(255, 233, 0), QColor(245, 175, 0), this);
	ui.horizontalLayout_header->addWidget(showFiltersButton);
	ui.horizontalLayout_header->addWidget(startOutputButton);
	*/
	showFilterString = "Show Filters";
	hideFilterString = "Hide Filters";

	startOutputString = "Start Output";
	stopOutputString = "Stop Output";

    ui.cameraList->setView(new QListView());

	isDraggingOver = false;
	isSurfaceSelected = false;

    //initCameraList();
	initCameraAndVideoFileList();

	timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(timerHandler()));
	//timer->start(0);
	//connect(timer, SIGNAL(timeout()), this, SLOT(processOneThing()));

#ifdef NDEBUG
	timer->start(16); // 60 FPS limit for release
#else
	timer->start(30); // 30 FPS limit for debug (to speed up UI)
#endif

	isVideoStarted = false;
	isOutputing = false;
	isFilterDisplaying = false;

	activeFilterButton = nullptr;
	bool loadCustom = false;
	boost::optional<boost::property_tree::ptree> filterListRecord;

	try // SAFETY LOAD FROM THE config.json
	{
		boost::property_tree::ptree config;

		boost::property_tree::read_json("./assets/application_config.json", config);

		bool createMode = config.get<bool>("editor", false);

		loadCustom = config.get<bool>("customConfig", false);

		if (!createMode)
		{
			//ui.create_filter->hide();
			//ui.edit_filter->hide();
			//ui.delete_filter->hide();
			//ui.import_filter->hide();
		}

		auto JSONsource = config.get<std::string>("filterConfig", "");

		if (!JSONsource.empty())
		{
			boost::property_tree::read_json(JSONsource, config);
		}
		else
		{
			throw std::exception();

			//boost::property_tree::read_json("./assets/fx_config/ui-config.json", config);

			//boost::property_tree::read_json("./assets/fx_config/ui-config-EDITOR.json", config);

			//boost::property_tree::read_json("./assets/fx_config/ui-config-TMP.json", config);

			//boost::property_tree::read_json("./assets/fx_config/ui-config-FULL-TEST.json", config);

		}

		filterListRecord = config.get_child_optional("filterConfig");

		if (filterListRecord)
		{
			for (auto &filterRecord : filterListRecord.get())
			{
				auto externalInfo = FilterUiModel(filterRecord.second);
				if (addFilter(externalInfo))
				{
					if (externalInfo.getTitle().empty())
					{
						externalInfo.setTitle(fs::path(externalInfo.getJSONsource()).filename().string());
					}

					filterUiModels.push_back(externalInfo);
				}
				else
				{
					qDebug() << "\n!!! FILTER LOADING ERROR: " << externalInfo.getTitle().c_str() << " | " << externalInfo.getId() << " | " << externalInfo.getJSONsource().c_str();
				}
			}
		}
	}
	catch (const std::exception&)
	{
		qDebug().noquote() << "\n!!! CONFIG LOADING ERROR !!!\n";
	}
	catch (...)
	{
		qDebug().noquote() << "\n!!! CONFIG LOADING ERROR !!!\n";
	}

	if (loadCustom)
	{
		// autoscan the CUSTOM folder & load filters
		for (auto& entry : fs::directory_iterator("./assets/fx_config/custom-filters"))
		{
			if (fs::is_directory(entry))
			{
				auto JSONsource = fs::path(entry.path() / "config.json");
				if (fs::exists(JSONsource))
				{
					FilterUiModel externalInfo;
					externalInfo.setJSONsource(JSONsource.string());

					if (addFilter(externalInfo))
					{
						if (externalInfo.getTitle().empty())
						{
							externalInfo.setTitle(fs::path(externalInfo.getJSONsource()).filename().string());
						}

						filterUiModels.push_back(externalInfo);
					}
					else
					{
						qDebug().noquote() << "\n!!! FILTER LOADING ERROR: \"" + QString::fromStdString(externalInfo.getTitle()) + "\"\n";
					}
				}
			}
		}
	}


	for (auto &externalInfo : filterUiModels)
	{
		filterManager.loadExternFilters(externalInfo, filterUiModels);
	}

	//if (ui.filterList->findChildren<QPushButton*>().count() == 0) // need one button for the editor, if none filter was loaded
	//{
	//	QPushButton *newbutton = new FilterButton(QString::fromStdString(externalInfo.getTitle()), QString::fromStdString(externalInfo.getIcon()), this);
	//	newbutton->setProperty("buttonNumber", 0);
	//	newbutton->setVisible(false);
	//	addFilterButton(newbutton);
	//}

	auto filter = filterManager.getCurrentFilter();

	if (filter != nullptr) // if current filter has modules, load their icons
	{
		if (filter->editable)
		{
			//ui.delete_filter->setEnabled(true);
		}

		for (int i = 0; i < filter->filterModules.size(); ++i)
		{
			QPushButton *button = qobject_cast<QPushButton*>(ui.horizontalLayout_moduleList->itemAt(i)->widget());
			button->setIcon(filter->filterModules[i].icon);
			button->setProperty("filterModule", (qulonglong)&filter->filterModules[i]);
			button->setVisible(true);
		}

		//ui.label_current_filter->setText(QString::fromStdString(filterUiModels.front().getTitle()));
	}

	connect(ui.moduleListLeft, SIGNAL(clicked()), this, SLOT(on_moduleListLeft_clicked()));
	connect(ui.moduleListRight, SIGNAL(clicked()), this, SLOT(on_moduleListRight_clicked()));

	connect(ui.trackFaceCheckBox, SIGNAL(stateChanged(int)), this, SLOT(on_trackFaces_stateChanged(int)));
	connect(ui.applyForAllCheckBox, SIGNAL(stateChanged(int)), this, SLOT(on_sameFiltersForAllFaces_stateChanged(int)));
	connect(ui.showFpsCheckBox, SIGNAL(stateChanged(int)), this, SLOT(on_showFps_stateChanged(int)));

	installEventFilters();
}

LiveCamWindow::~LiveCamWindow(){
}

bool LiveCamWindow::addFilter(FilterUiModel& externalInfo)
{
	std::string JSONsource = externalInfo.getJSONsource();

	FilterButton *filterButton = new FilterButton(QString::fromStdString(externalInfo.getTitle()), QString::fromStdString(externalInfo.getIcon()), this);
	//externalInfo.connectedButton = filterButton;

	try
	{
		if (JSONsource.empty()) // CHOICE BETWEEN HARDCODED AND TEMPLATED FILTER
		{
			if (externalInfo.getId() == 0)
			{
				throw std::exception();
			}
			else
			{
				filterManager.addFilter(externalInfo.getId());
			}
		}
		else
		{
			if (filterManager.addFilter(&externalInfo) == 0)
			{
				throw std::exception();
			}
		}
	}
	catch (std::exception&)
	{
		return false;
	}
	catch (...)
	{
		return false;
	}

	int	number = ui.filterList->findChildren<FilterButton*>().count();

	filterButton->setProperty("buttonNumber", number);
	addFilterButton(filterButton);
	// icon setting is below

	if (number == 0)
	{
		filterButton->setChecked(true);
		activeFilterButton = filterButton;
	}

	auto &filter = filterManager.findFilterById(externalInfo.getId());

	// ICON SETTING
	if (!externalInfo.getIcon().empty())
	{
		filterButton->setIcon(QIcon(QString::fromStdString(fs::path(filter->resourcesRoot / externalInfo.getIcon()).string())));
	}

	int count = ui.moduleList->findChildren<QPushButton*>().count();
	count = filter->filterModules.size() - count;

	while (count-- > 0)
	{
		addModuleButton();
	}

	return true;
}

void LiveCamWindow::addFilter(bool rewrite)
{
	auto editorData = editorWindow->getEditableFilterExtraData();

	FilterButton *filterButton;
	int number;

	if (activeFilterButton)
	{
		if (rewrite)
		{
			number = activeFilterButton->property("buttonNumber").toInt();
			filterButton = activeFilterButton;
		}
		else
		{
			number = ui.filterList->findChildren<FilterButton*>().count();
			filterButton = new FilterButton(QString::fromStdString(editorData.title), QString::fromStdString(editorData.filterIconPath), this);
			addFilterButton(filterButton);
		}
	}
	else
	{
		number = 0;
		filterButton = (FilterButton*)ui.horizontalLayout_filterList->itemAt(0)->widget();
	}

	if (!rewrite)
	{
		filterButton->setProperty("buttonNumber", number);
	}

	if (number == 0)
	{
		filterButton->setChecked(true);
		activeFilterButton = filterButton;
	}

	FilterUiModel newExternalData;
	//newExternalData.connectedButton = filterButton;
	newExternalData.setTitle(editorData.title);
	newExternalData.setIcon(fs::path(editorData.filterIconPath).filename().string());

	filterButton->setIcon(QIcon(editorData.filterIcon));

	if (!rewrite)
	{
		int templatedID = filterManager.addFilter(editorWindow->getEditableFilter());
		newExternalData.setId(templatedID);

		filterUiModels.push_back(newExternalData);
	}
	else
	{
		filterManager.filters[filterManager.getCurrentFilterID()] = editorWindow->editableFilter;

		newExternalData.setId(filterUiModels[number].getId());
		filterUiModels[number] = newExternalData;
	}
}


void LiveCamWindow::setCompositeLeftWidget(CVImageWidget* imageWidget)
{
	compositeLeftWidget = imageWidget;

	imageWidget->setText("No signal");
	imageWidget->setFont(QFont("Segoe UI Semibold", 20));
	imageWidget->setFixedSize(600, 300);
	imageWidget->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	imageWidget->setAlignment(Qt::AlignmentFlag::AlignHCenter | Qt::AlignmentFlag::AlignVCenter);
	imageWidget->setStyleSheet("background-image: url(:/images/NoSignal.png);\nbackground-repeat: repeat-xy;\n"
		"border-width: 4;\nborder-color: rgb(36, 46, 104);\nborder-style: solid;\ncolor: rgb(60, 60, 60);");

	imageWidget->setAcceptDrops(true);
	imageWidget->installEventFilter(this);

    ui.gridLayout_central->addWidget(imageWidget);
}

void LiveCamWindow::closeEvent(QCloseEvent*)
{
	closeCameras();
}

void LiveCamWindow::on_cameraList_currentIndexChanged(int index)
{
	isVideoStarted = index != 0;
	std::string cameraName = ui.cameraList->currentText().toStdString();

	index -= 1; // - 1, BECAUSE ITEM WITH 0 INDEX IS "None"

	if (isVideoStarted)
	{
		if (index < QCameraInfo::availableCameras().size())
		{
			startCamera(index);
		}
		else
		{
			startVideoFile(cameraName);
		}
	}

	if (filterManager.getCurrentFilter())
	{
		ui.showFiltersButton->setEnabled(isVideoStarted);
	}

	ui.startOutputButton->setEnabled(isVideoStarted);
}

void LiveCamWindow::on_showFiltersButton_clicked()
{
	if (!filterManager.getCurrentFilter())
	{
		return;
	}

	if (!isFilterDisplaying)
	{
		isFilterDisplaying = true;
	}
	else
	{
		isFilterDisplaying = false;
	}
}

void LiveCamWindow::initVideoFileList()
{
	using namespace boost::filesystem;

	path p("./videos/");

	if (is_directory(p)) {
		std::cout << p << " is a directory containing:\n";

		for (auto& entry : boost::make_iterator_range(directory_iterator(p), {}))
		{
			QString qstring = QString::fromStdString(entry.path().filename().generic_string());

			ui.cameraList->addItem(qstring);
		}

	}
}

void LiveCamWindow::on_startOutputButton_clicked() {
	if (!isOutputing)
	{
		isOutputing = true;
	}
	else
	{
		isOutputing = false;
	}
}

void LiveCamWindow::runTracker2(){

}

void LiveCamWindow::timerHandler(){

	static bool INIT_SCROLLBAR_STYLESHEETS = true;
	if (INIT_SCROLLBAR_STYLESHEETS) // a workaround to fix scrollbar rendering
	{
		INIT_SCROLLBAR_STYLESHEETS = false;
		ui.filterListScroll->setStyleSheet("#filterListScroll{\nbackground-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 rgb(10, 20, 40), stop:1 rgb(33, 43, 98));\n}\n\nQLabel {\nfont-family: \"Segoe UI Semibold\";\nfont-size: 14px;\ncolor: white;\n}\n\nQScrollBar{\nbackground-color:  rgb(20, 40, 80);\n}\n\nQScrollBar::handle:horizontal {\n    background-color: white;\nmargin: 3px;\nborder-radius: 5px;\n}\n\nQScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {\nbackground-color:  rgb(20, 40, 80);\n}\n\nQScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {\nborder-style: none;\n}\n\nQScrollBar::handle:hover {\n    background-color: rgb(35, 162, 255);\n}");
		ui.moduleListScroll->setStyleSheet("QScrollBar{\nbackground-color:  rgb(20, 40, 80);\n}\n\nQScrollBar::handle:horizontal {\n    background-color: white;\nmargin: 3px;\nborder-radius: 5px;\n}\n\nQScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {\nbackground-color:  rgb(20, 40, 80);\n}\n\nQScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {\nborder-style: none;\n}\n\nQScrollBar::handle:hover {\n    background-color: rgb(35, 162, 255);\n}");
		ui.cameraList->setStyleSheet("QComboBox {\nbackground: black;\nborder-top-right-radius: 5px;\nborder-bottom-right-radius: 5px;\ncolor: rgb(35, 162, 255);\n}\n\nQComboBox::drop-down {\nimage: url(:/images/ComboBoxWhiteArrow.png);\nimage-position: left;\nwidth: 20px;\nheight: 20px;\npadding: 5px;\nborder-width: 0px;\nsubcontrol-position: center right;\n}\n\nQComboBox QAbstractItemView {\n	background: black;\n	color: rgb(35, 162, 255);\n}\n\nQComboBox QAbstractItemView::item:selected {\n	color:  rgb(0, 210, 255);\n}");
		ui.filterList->setMinimumHeight(130); // to fix vertical layout
		ui.filterListContainer->setMinimumHeight(130); // to fix vertical layout
		ui.filterListScroll->setMinimumHeight(150); // to fix vertical layout
	}

	runTracker();
}

void LiveCamWindow::stopTimer(){
    timer->stop();
}

void LiveCamWindow::setRefresh(double fpsLimit) {
#ifdef NDEBUG
	int refreshRate = round(1 / fpsLimit * 1000);
	timer->start(refreshRate);
#else
	timer->start(30); // 30 FPS limit for debug (to speed up UI)
#endif
}

void LiveCamWindow::initCameraAndVideoFileList() {

	initCameraList();
	initVideoFileList();
}

void LiveCamWindow::initCameraList(){
    // Getting devices list and adding to UI.
    QActionGroup *videoDevicesGroup = new QActionGroup(this);
    videoDevicesGroup->setExclusive(false);
    foreach(const QCameraInfo &cameraInfo, QCameraInfo::availableCameras()) {
        QAction *videoDeviceAction = new QAction(cameraInfo.description(), videoDevicesGroup);
        videoDeviceAction->setCheckable(true);
        videoDeviceAction->setData(QVariant::fromValue(cameraInfo));
        if (cameraInfo == QCameraInfo::defaultCamera()){
            videoDeviceAction->setChecked(true);
        }
				QString camDesc = cameraInfo.description();
        ui.cameraList->addItem(camDesc);
				std::string utf8_camDesc = camDesc.toUtf8().constData();
				cameraNames.push_back(utf8_camDesc);
    }

    if (QCameraInfo::availableCameras().length()==0){
		cv::VideoCapture camera;
        int device_counts = 0;
        while (true) {
            if (!camera.open(device_counts++)) {
                break;
            }
            QString qstring;
            QTextStream(&qstring) << "Camera " << device_counts;
            ui.cameraList->addItem(qstring);
        }
        camera.release();
    }
}

void LiveCamWindow::on_actionExit_triggered(){
	closeCameras();
}

void LiveCamWindow::displayFrame(cv::Mat3b frame, int index){
	rawLeftWidget->showImage(frame);
	outputImageWidget->showImage(frame);
}

void LiveCamWindow::setFilter(std::shared_ptr<FX> filter)
{
	sendEffectToGraphics(filter);
}

void LiveCamWindow::setFilter(int value){
	if (filterUiModels.size() < value + 1) return;

	QString title = QString::fromUtf8(filterUiModels[value].getTitle().c_str());

	setWatermark(filterUiModels[value].getWatermark());
	loadFilter(filterUiModels[value].getId());

	animatedSubmenuReplacing(filterManager.getCurrentFilter());
}

void LiveCamWindow::on_filterButton_clicked(FilterButton* sender)
{
	qDebug() << "button " << sender->property("buttonNumber").toString();
	if (activeFilterButton == sender)
	{
		return;
	}
	activeFilterButton = sender;
	setFilter(sender->property("buttonNumber").toInt());
}

void LiveCamWindow::on_moduleListLeft_clicked()
{
	ui.moduleListScroll->horizontalScrollBar()->setValue(ui.moduleListScroll->horizontalScrollBar()->value() - 100);

}


void LiveCamWindow::on_moduleListRight_clicked()
{
	ui.moduleListScroll->horizontalScrollBar()->setValue(ui.moduleListScroll->horizontalScrollBar()->value() + 100);
}

void LiveCamWindow::on_trackFaces_stateChanged(int state)
{
	if (state == Qt::Checked)
	{
		detector.start();
		showComposite(true, filterManager.getCurrentFilter());
	}
	else if (state == Qt::Unchecked)
	{
		detector.stop();
		showComposite(false, nullptr);
	}
}

void LiveCamWindow::on_sameFiltersForAllFaces_stateChanged(int state)
{
	if (state == Qt::Checked)
	{
		setSameFiltersForAllFaces(true);
	}
	else if (state == Qt::Unchecked)
	{
		setSameFiltersForAllFaces(false);
	}
}

void LiveCamWindow::on_showFps_stateChanged(int state)
{
	if (state == Qt::Checked)
	{
		showFps();
	}
	else if (state == Qt::Unchecked)
	{
		hideFps();
	}
}





bool LiveCamWindow::eventFilter(QObject *obj, QEvent *event)
{
    if (event->type() == QEvent::MouseButtonPress ) {
        QMouseEvent *mouse_event = static_cast<QMouseEvent *>(event);
        if (mouse_event->button() != Qt::LeftButton) return false;
		if (detector.activeTrackingType == TRACKER_TYPE::SURFACE && !isSurfaceSelected)
		{
			isSurfaceSelected = true;
			QPoint mxy = QWidget::mapFromGlobal(QCursor::pos());
			int x = mxy.x();
			int y = mxy.y();
			takeSnapshot(x, y);
		}
		else
		{
			if (qobject_cast<QPushButton *>(obj) == nullptr) return false;

			qDebug() << mouse_event;

			mouse_event->pos();

			QPushButton* button = qobject_cast<QPushButton*>(obj);

			QPixmap pixmap = QPixmap(button->icon().pixmap(QSize(60, 60)));

			QDrag *drag = new QDrag(this);
			QMimeData *mimeData = new QMimeData;

			mimeData->setImageData(pixmap);
			mimeData->setProperty("filterModule", button->property("filterModule"));

			drag->setMimeData(mimeData);
			drag->setHotSpot(QPoint(0, 0));
			drag->setPixmap(pixmap);

			drag->exec();
		}
    }

    if (event->type() == QEvent::MouseButtonDblClick ) {
        QMouseEvent *mouse_event = static_cast<QMouseEvent *>(event);
        if (mouse_event->button() != Qt::LeftButton) return false;
        if (qobject_cast<QPushButton *>(obj) == nullptr) return false;

        qDebug() << mouse_event;

		doubleClickSelection((FilterModule*)obj->property("filterModule").toULongLong());
    }

    if (event->type() == QEvent::DragEnter) {
        qDebug() << "QEvent::DragEnter";
		isDraggingOver = true;
		usingDragDrop(isDraggingOver);
        QDragEnterEvent *drag_enter_event = static_cast<QDragEnterEvent *>(event);
        drag_enter_event->acceptProposedAction();
    }

	if (isDraggingOver){
		if (event->type() == QEvent::DragMove) {
			QDragMoveEvent *drag_move_event = static_cast<QDragMoveEvent *>(event);
			qDebug() << "QEvent::DragMove" << drag_move_event->pos();
			drag_move_event->acceptProposedAction();
			draggingOver(drag_move_event->pos());
		}
	}

	if (event->type() == QEvent::DragLeave) {
		qDebug() << "QEvent::DragLeave";
		isDraggingOver = false;
		usingDragDrop(false);
	}

    if (event->type() == QEvent::Drop) {

		int index = obj->property("moduleID").toInt();

        QDropEvent *drop_event = static_cast<QDropEvent *>(event);
        drop_event->acceptProposedAction();

		auto s = drop_event->mimeData()->property("filterModule").toULongLong();
		dropImage((FilterModule*)drop_event->mimeData()->property("filterModule").toULongLong());

		isDraggingOver = false;
		usingDragDrop(false);
    }

    return false;
}

void LiveCamWindow::installEventFilters()
{

}

void LiveCamWindow::onCamEventAction(QString camFileName, int index)
{

}

void LiveCamWindow::animatedSubmenuReplacing(std::shared_ptr<FX> newSettedFilter)
{
	QHBoxLayout *filterModulesTableGrid = findChild<QHBoxLayout*>("horizontalLayout_moduleList");
	QWidget *filterModulesTable = filterModulesTableGrid->parentWidget();

	QGraphicsBlurEffect *blur1 = new QGraphicsBlurEffect();
	QPropertyAnimation *blurIncreasing = new QPropertyAnimation(blur1, "blurRadius");
	blurIncreasing->setStartValue(1);
	blurIncreasing->setEndValue(10);
	blurIncreasing->setDuration(100);

	QGraphicsBlurEffect *blur2 = new QGraphicsBlurEffect();
	QPropertyAnimation* blurDecreasing = new QPropertyAnimation(blur2, "blurRadius");
	blurDecreasing->setStartValue(10);
	blurDecreasing->setEndValue(1);
	blurDecreasing->setDuration(100);

	filterModulesTable->setGraphicsEffect(blur1);
	blurIncreasing->start();

    connect(blurIncreasing, &QPropertyAnimation::finished, [=]()
	{
		int i = 0;

		if (newSettedFilter)
		{
			for (; i < newSettedFilter->filterModules.size(); ++i)
			{
				QPushButton *button = qobject_cast<QPushButton*>(filterModulesTableGrid->itemAt(i)->widget());
				button->setIcon(newSettedFilter->filterModules[i].icon);
				button->setVisible(true);
				button->setProperty("filterModule", (qulonglong)&newSettedFilter->filterModules[i]);
			}
		}

		QLayoutItem *item;

		while ((item = filterModulesTableGrid->itemAt(i)) != nullptr)
		{
			item->widget()->setVisible(false);
			((QPushButton*)item->widget())->setIcon(QIcon());
			++i;
		}

		filterModulesTable->setGraphicsEffect(blur2);
		blurDecreasing->start();
	});
}

void LiveCamWindow::animatedMainMenuReplacing(bool swapToEditor)
{
	QGridLayout *filterTableGrid = findChild<QGridLayout*>("gridLayout_filters_content");
	QWidget *filterTable = filterTableGrid->parentWidget();

	QGraphicsBlurEffect *blur1 = new QGraphicsBlurEffect();
	QPropertyAnimation *blurIncreasing = new QPropertyAnimation(blur1, "blurRadius");
	blurIncreasing->setStartValue(1);
	blurIncreasing->setEndValue(10);
	blurIncreasing->setDuration(100);

	QGraphicsBlurEffect *blur2 = new QGraphicsBlurEffect();
	QPropertyAnimation* blurDecreasing = new QPropertyAnimation(blur2, "blurRadius");
	blurDecreasing->setStartValue(10);
	blurDecreasing->setEndValue(1);
	blurDecreasing->setDuration(100);

	filterTable->setGraphicsEffect(blur1);
	blurIncreasing->start();

	connect(blurIncreasing, &QPropertyAnimation::finished, [=]()
	{
		if (swapToEditor)
		{
			bool firstButton = true;
			for (int i = 0; i < filterUiModels.size() || firstButton; ++i)
			{
				QPushButton *button = qobject_cast<QPushButton*>(filterTableGrid->itemAt(i)->widget());
				if (firstButton)
				{
					QIcon icon;
					icon.addPixmap(QPixmap(QString::fromStdString(editorWindow->data.filterIconPath)), QIcon::Disabled);
					button->setIcon(icon);
					button->setVisible(true);
					firstButton = false;

					if (activeFilterButton)
					{
						activeFilterButton->setChecked(false);
					}
				}
				else
				{
					button->setVisible(false);
				}
			}
		}
		else
		{
			if (filterUiModels.size() == 0)
			{
				((QPushButton*)filterTableGrid->itemAt(0)->widget())->setVisible(false);
			}
			else
			{
				bool firstButton = true;
				for (int i = 0; i < filterUiModels.size() || firstButton; ++i)
				{
					QPushButton *button = qobject_cast<QPushButton*>(filterTableGrid->itemAt(i)->widget());
					if (firstButton)
					{
						std::string iconPath = boost::filesystem::path(
							filterManager.findFilterById(filterUiModels.front().getId())->resourcesRoot / filterUiModels.front().getIcon()
						).string();

						button->setIcon(QIcon(QString::fromStdString(iconPath)));

						if (activeFilterButton)
						{
							activeFilterButton->setChecked(true);
						}

						firstButton = false;
					}
					else
					{
						button->setVisible(true);
					}
				}
			}
		}

		filterTable->setGraphicsEffect(blur2);
		blurDecreasing->start();
	});
}

void LiveCamWindow::on_create_filter_clicked()
{

}

void LiveCamWindow::on_editor_filterIcon_changed()
{

}

void LiveCamWindow::on_editor_moduleIcon_changed()
{

}

void LiveCamWindow::on_edit_filter_clicked()
{

}

void LiveCamWindow::on_editor_window_closed()
{

}

void LiveCamWindow::on_delete_filter_clicked()
{

}

void LiveCamWindow::on_import_filter_clicked()
{
	auto JSONsource = QFileDialog::getOpenFileName(nullptr, tr("Choose a Filter Config to Load"),
		editorWindow->filterFolderStart, "filter-config.json");

	if (JSONsource.isEmpty())
	{
		return;
	}

	editorWindow->filterFolderStart = QFileInfo(JSONsource).dir().path();

	FilterUiModel externalInfo;
	externalInfo.setJSONsource(JSONsource.toStdString());

	addFilter(externalInfo);
	filterUiModels.push_back(externalInfo);

	auto filterFolder = fs::path(JSONsource.toStdString()).parent_path();
	fs::copy_directory(filterFolder, "./assets/fx_config/custom-filters/" / filterFolder.filename());
}

void LiveCamWindow::on_maxTargetSlider_valueChanged(int value)
{
	ui.maxsCounterLabel->setText(QString::number(value));
	detector.setTargetLimit(value);
}


bool LiveCamWindow::areFacesTracked()
{
	return ui.trackFaceCheckBox->isChecked();
}
