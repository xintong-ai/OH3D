
//#include "glextensions.h"
#include <string>
#include <QApplication>
#include <window.h>
//CHANGE_Huijie
//#include <GL/freeglut.h>

//inline bool matchString(const char *extensionString, const char *subString)
//{
//    int subStringLength = strlen(subString);
//    return (strncmp(extensionString, subString, subStringLength) == 0)
//        && ((extensionString[subStringLength] == ' ') || (extensionString[subStringLength] == '\0'));
//}
//
//bool necessaryExtensionsSupported()
//{
//    const char *extensionString = reinterpret_cast<const char *>(glGetString(GL_EXTENSIONS));
//    const char *p = extensionString;
//
//    const int GL_EXT_FBO = 1;
//    const int GL_ARB_VS = 2;
//    const int GL_ARB_FS = 4;
//    const int GL_ARB_SO = 8;
//    int extensions = 0;
//
//    while (*p) {
//        if (matchString(p, "GL_EXT_framebuffer_object"))
//            extensions |= GL_EXT_FBO;
//        else if (matchString(p, "GL_ARB_vertex_shader"))
//            extensions |= GL_ARB_VS;
//        else if (matchString(p, "GL_ARB_fragment_shader"))
//            extensions |= GL_ARB_FS;
//        else if (matchString(p, "GL_ARB_shader_objects"))
//            extensions |= GL_ARB_SO;
//        while ((*p != ' ') && (*p != '\0'))
//            ++p;
//        if (*p == ' ')
//            ++p;
//    }
//    return (extensions == 15);
//}

int main(int argc, char **argv)
{
	QApplication app(argc, argv);

	Window win;
	//    win.setWindowState(win.windowState() ^ Qt::WindowFullScreen);

	win.show();

	return app.exec();
}


//#include <QApplication>
//#include <QDesktopWidget>
//
//#include "window.h"
//
//int main(int argc, char *argv[])
//{
//	QApplication app(argc, argv);
//	Window window;
//	window.resize(window.sizeHint());
//	int desktopArea = QApplication::desktop()->width() *
//		QApplication::desktop()->height();
//	int widgetArea = window.width() * window.height();
//	if (((float)widgetArea / (float)desktopArea) < 0.75f)
//		window.show();
//	else
//		window.showMaximized();
//	return app.exec();
//}
