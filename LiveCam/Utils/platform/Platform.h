#ifndef LIVECAM_PLATFORMH
#define LIVECAM_PLATFORMH

// Info on predefs http://sourceforge.net/apps/mediawiki/predef/index.php?title=Operating_Systems

// Systems
#define LIVECAM_WINDOWS 1
#define LIVECAM_APPLE 2
#define LIVECAM_IOS 3
#define LIVECAM_LINUX 4

// Creative libs/GL-helpers, used for e.g. data paths
#define LIVECAM_GLFW 1
#define LIVECAM_OPENFRAMEWORKS 2
#define LIVECAM_CINDER 3
#define LIVECAM_COCOA 4


// Detect system
#ifdef _WIN32
#define LIVECAM_PLATFORM LIVECAM_WINDOWS
#elif __linux__
#define LIVECAM_PLATFORM LIVECAM_LINUX
#elif __APPLE__
 #if TARGET_OS_IPHONE    
   #define LIVECAM_PLATFORM  LIVECAM_IOS
 #else
   #define LIVECAM_PLATFORM  LIVECAM_APPLE
 #endif
#else
#error Unsupported operating system
#endif


// Include platform specifics
#if LIVECAM_PLATFORM == LIVECAM_WINDOWS
#include <Utils/platform/windows/WindowsPlatform.h>
#elif LIVECAM_PLATFORM == LIVECAM_APPLE || LIVECAM_PLATFORM == LIVECAM_IOS
#include <Utils/platform/osx/OSXPlatform.h>
#elif LIVECAM_PLATFORM == LIVECAM_LINUX
#include <Utils/platform/linux/LinuxPlatform.h>
#endif

#endif
