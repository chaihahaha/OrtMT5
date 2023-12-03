# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.13)

# Project
project(OrtMT5 C CXX)
string(APPEND CMAKE_CXX_FLAGS " /W4")

if(NOT ONNXRUNTIME_ROOTDIR)
  set(ONNXRUNTIME_ROOTDIR "D:/libs/onnxruntime-win-x64-1.16.2")
endif()
if(NOT SENTENCEPIECE_ROOTDIR)
    set(SENTENCEPIECE_ROOTDIR "D:/libs/sentencepiece-0.1.99-win64")
endif()
if(NOT ARGPARSE_ROOTDIR)
    set(ARGPARSE_ROOTDIR "D:/libs/argparse")
endif()
include_directories("${ONNXRUNTIME_ROOTDIR}/include" "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")
include_directories("${SENTENCEPIECE_ROOTDIR}/include")
include_directories("${ARGPARSE_ROOTDIR}/include")
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")
link_directories("${SENTENCEPIECE_ROOTDIR}/lib")
find_package(Threads) 

#if(MSVC)
#    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
#    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_RELEASE} /MTd")
#endif()

if (CMAKE_BUILD_TYPE STREQUAL Debug)
    ADD_DEFINITIONS(-DDEBUG)
endif()
if (CMAKE_BUILD_TYPE STREQUAL Release)
    ADD_DEFINITIONS(-DRELEASE)
endif()

add_executable(ortmt5 "OrtMT5.cpp" "OrtMT5.h")

target_link_libraries(ortmt5 PRIVATE onnxruntime sentencepiece Threads::Threads)
target_compile_features(ortmt5 PRIVATE cxx_std_17)
#In onnxruntime deafault install path, the required dlls are in lib and bin folders
set(DLL_DIRS "${ONNXRUNTIME_ROOTDIR}/lib;${ONNXRUNTIME_ROOTDIR}/bin;${SENTENCEPIECE_ROOTDIR}/lib;${SENTENCEPIECE_ROOTDIR}/bin;")
foreach(DLL_DIR IN LISTS DLL_DIRS)
  file(GLOB ALL_DLLS ${DLL_DIR}/*.dll)
  foreach(ORTDll IN LISTS ALL_DLLS)
    add_custom_command(TARGET ortmt5 POST_BUILD 
				COMMAND ${CMAKE_COMMAND} -E copy_if_different
				"${ORTDll}"              
				$<TARGET_FILE_DIR:ortmt5>)
  endforeach()
endforeach()
