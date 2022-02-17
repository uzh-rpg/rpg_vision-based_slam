/* This file is part of RPG vision-based SLAM.
Copyright (C) 2022 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).

This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
*/

/*
This script has been adapted from: 
https://gitlab.com/VladyslavUsenko/basalt-headers/-/blob/master/include/basalt/utils/assert.h
*/

#pragma once

#include <iostream>

namespace gvi_fusion {


#define UNUSED(x) (void)(x)

inline void assertion_failed(char const* expr, char const* function,
                             char const* file, long line) {
  std::cerr << "***** Assertion (" << expr << ") failed in " << function
            << ":\n"
            << file << ':' << line << ":" << std::endl;
  std::abort();
}

inline void assertion_failed_msg(char const* expr, char const* msg,
                                 char const* function, char const* file,
                                 long line) {
  std::cerr << "***** Assertion (" << expr << ") failed in " << function
            << ":\n"
            << file << ':' << line << ": " << msg << std::endl;
  std::abort();
}
}  // namespace gvi_fusion

/*#if defined(BASALT_DISABLE_ASSERTS)

#define BASALT_ASSERT(expr) ((void)0)

#define BASALT_ASSERT_MSG(expr, msg) ((void)0)

#define BASALT_ASSERT_STREAM(expr, msg) ((void)0)

#else

#define BASALT_ASSERT(expr)                                               \
  (BASALT_LIKELY(!!(expr))                                                \
       ? ((void)0)                                                        \
       : ::basalt::assertion_failed(#expr, __PRETTY_FUNCTION__, __FILE__, \
                                    __LINE__))

#define BASALT_ASSERT_MSG(expr, msg)                                     \
  (BASALT_LIKELY(!!(expr))                                               \
       ? ((void)0)                                                       \
       : ::basalt::assertion_failed_msg(#expr, msg, __PRETTY_FUNCTION__, \
                                        __FILE__, __LINE__))*/

#define GVI_FUSION_LIKELY(x) __builtin_expect(x, 1)

#define GVI_FUSION_ASSERT_STREAM(expr, msg)                                    \
  (GVI_FUSION_LIKELY(!!(expr))                                                 \
       ? ((void)0)                                                         \
       : (std::cerr << msg << std::endl,                                   \
          ::gvi_fusion::assertion_failed(#expr, __PRETTY_FUNCTION__, __FILE__, \
                                     __LINE__)))

//#endif

