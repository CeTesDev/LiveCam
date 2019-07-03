#pragma once

#include <QMainWindow>
#include <QMouseEvent>
#include <QEvent>
#include <QFileInfo>
#include <QFile>
#include <QDebug>
#include <QPixmap>
#include <QFileDialog>
#include "ui_EditorWindow.h"

#include <modelTabs/EmojiModelTab.h>
#include <modelTabs/Model3DTab.h>

#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <Widgets/FxWidget3D.h>
#include <Widgets/FxWidget2D.h>
#include <Widgets/FxWidget2DAnimated.h>
#include <Widgets/FxWidgetFacePaint.h>
#include <Widgets/FxModDepthMask.h>

QT_BEGIN_NAMESPACE
namespace Ui { class EditorWindow; }
QT_END_NAMESPACE

class LiveCamWindow;

class EditorWindow : public QMainWindow
{
    Q_OBJECT

public:
	EditorWindow(LiveCamWindow *liveCamWindow, QWidget *parent = Q_NULLPTR);
    ~EditorWindow();

	void resetEditableFilter();

	std::shared_ptr<FX> getEditableFilter();
	ExtraFilterData getEditableFilterExtraData();

	void readFilterProperties(std::shared_ptr<FX> filter, FilterUiModel& externalInfo);

	EmojiModelTab* createEmojiModelTab();
	Model3DTab* EditorWindow::createModel3DTab(bool lipsJoint);

public: signals:
	void editor_window_closed();

public slots:

	void on_tab_deleted();
	void on_filterName_edit_textEdited(const QString &text);
	void on_filterIcon_button_clicked();
	void on_filterIcon_delete_clicked();

	void on_new_emoji_clicked();
	void on_new_3D_clicked();
	void on_new_LipsJoint_clicked();

	void on_models_tabWidget_currentChanged(int index);

	void on_hide_button_clicked();
	void on_save_button_clicked();
	void on_export_button_clicked();
	void on_import_button_clicked();

private:

	friend class LiveCamWindow;
	friend class EmojiModelTab;
	friend class Model3DTab;

	QIcon chooseImageIcon;

	QString fileDialogStart;
	QString filterFolderStart;

    Ui::EditorWindow ui;
	LiveCamWindow* liveCamWindow;

	ExtraFilterData data;
	std::shared_ptr<FX> editableFilter;

	bool EditorWindow::event(QEvent * e) override;

	QPushButton* createIconButton();
	void saveFilter(QString& filterFolder);

	void setFilterIcon(QString& iconPath);
	void setElementIcon(QString& iconPath);

protected:
	void closeEvent(QCloseEvent *event) override;
};

#include <LiveCamWindow.h>