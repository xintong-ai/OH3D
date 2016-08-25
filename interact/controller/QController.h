#pragma once
#include <QObject>
#include <QVector3D>

class Controller;

//#ifdef EXPORT_QT
//#define TEST_COMMON_DLLSPEC Q_DECL_EXPORT
//#else
//#define TEST_COMMON_DLLSPEC Q_DECL_IMPORT
//#endif

class /*TEST_COMMON_DLLSPEC*/  QController :public QObject
{
	Q_OBJECT


private:
	Controller *controller;
public:
	QController();

signals:
	void SignalUpdateControllers(QVector3D leftPos, QVector3D rightPos, 
		int numHands, bool pressed);
public slots:
	void Update();

};