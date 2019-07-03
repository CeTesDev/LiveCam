#ifndef LIVECAM_LOG_H
#define LIVECAM_LOG_H

#include <iostream>
#include <stdarg.h>

#define LC_LOG_NO_COLOR
#define LC_LOG_LEVEL_ALL 4
#define LC_LOG_LEVEL_ERROR  1
#define LC_LOG_LEVEL_WARNING 2
#define LC_LOG_LEVEL_VERBOSE 3

#define LC_LOG_LEVEL_NONE 0 

extern "C" {

typedef void(*roxlu_log_callback)(int level, void* user, int line, const char* function, const char* fmt, va_list args);

extern roxlu_log_callback roxlu_log_cb;
extern void* roxlu_log_user;
extern int roxlu_log_level;

void lc_log_set_callback(roxlu_log_callback cb, void* user);
void lc_log_set_level(int level); 
void lc_log_default_callback(int level, void* user, int line, const char* function, const char* fmt, va_list args); 

void lc_verbose(int line, const char* function, const char* fmt, ...);
void lc_warning(int line, const char* function, const char* fmt, ...);
void lc_error(int line, const char* function, const char* fmt, ...);

#define LC_VERBOSE(fmt, ...) { lc_verbose(__LINE__, __FUNCSIG__, fmt, ##__VA_ARGS__); } 
#define LC_WARNING(fmt, ...) { lc_warning(__LINE__, __FUNCSIG__, fmt, ##__VA_ARGS__); } 
#define LC_ERROR(fmt, ...) { lc_error(__LINE__, __FUNCSIG__, fmt, ##__VA_ARGS__); } 

#ifndef LC_LOG_LEVEL
#define LC_LOG_LEVEL LC_LOG_LEVEL_VERBOSE
#endif

// unset log macros based on the -DLC_LOG_LEVEL preprocessor variable
#if LC_LOG_LEVEL == LC_LOG_LEVEL_VERBOSE
#elif LC_LOG_LEVEL ==  LC_LOG_LEVEL_WARNING
  #undef LC_VERBOSE
#elif LC_LOG_LEVEL == LC_LOG_LEVEL_ERROR
  #undef LC_VERBOSE
  #undef LC_WARNING
#endif

#if LC_LOG_LEVEL == LC_LOG_LEVEL_NONE
  #undef LC_ERROR
  #undef LC_VERBOSE
  #undef LC_WARNING
#endif

// if not set anymore, define empties
#ifndef LC_VERBOSE
  #define LC_VERBOSE(fmt, ...) {}
#endif

#ifndef LC_WARNING
  #define LC_WARNING(fmt, ...) {}
#endif

#ifndef LC_ERROR
  #define LC_ERROR(fmt, ...) {}
#endif
} // extern C
#endif // LIVECAM_LOG_H

