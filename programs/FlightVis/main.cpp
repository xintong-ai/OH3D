
//#include "glextensions.h"
#include <string>
#include <QApplication>
#include <QGLFormat>
#include <window.h>

int main(int argc, char **argv)
{
	//using the following statement, 
	//all the OpenGL contexts in this program are shared.
	QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
	QApplication app(argc, argv);




	// QSurfaceFormat format;
	// format.setProfile(QSurfaceFormat::CompatibilityProfile);
	// format.setVersion(4, 1);
	// QSurfaceFormat::setDefaultFormat(format);

	// QSurfaceFormat format;
 //    format.setDepthBufferSize(24);
 //    QSurfaceFormat::setDefaultFormat(format);

	// QGLFormat glFormat;
	// glFormat.setVersion( 4, 1 );
	// glFormat.setProfile( QGLFormat::CoreProfile ); // Requires >=Qt-4.8.0
	// glFormat.setSampleBuffers( true );
	// QSurfaceFormat::setDefaultFormat(glFormat);

    // QSurfaceFormat format;
    // format.setDepthBufferSize( 24 );
    // format.setMajorVersion( 3 );
    // format.setMinorVersion( 2 );
    // format.setSamples( 4 );
    // format.setProfile( QSurfaceFormat::CoreProfile );
    // QSurfaceFormat::setDefaultFormat( format );

	Window win;
	win.show();
	win.init();
	//win.setFormat(format);

	return app.exec();
}
