# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.13)

# Project
project(OrtMT5 C CXX)
string(APPEND CMAKE_CXX_FLAGS " /W4")

if(NOT ONNXRUNTIME_ROOTDIR)
  set(ONNXRUNTIME_ROOTDIR "D:/libs/onnxruntime-win-x64-1.16.2")
endif()
include_directories("${ONNXRUNTIME_ROOTDIR}/include" "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

add_executable(ortmt5 "OrtMT5.cpp")
target_link_libraries(ortmt5 PRIVATE onnxruntime)
target_compile_features(ortmt5 PRIVATE cxx_std_17)

#In onnxruntime deafault install path, the required dlls are in lib and bin folders
set(DLL_DIRS "${ONNXRUNTIME_ROOTDIR}/lib;${ONNXRUNTIME_ROOTDIR}/bin")
foreach(DLL_DIR IN LISTS DLL_DIRS)
  file(GLOB ALL_DLLS ${DLL_DIR}/*.dll)
  foreach(ORTDll IN LISTS ALL_DLLS)
    add_custom_command(TARGET ortmt5 POST_BUILD 
				COMMAND ${CMAKE_COMMAND} -E copy_if_different
				"${ORTDll}"              
				$<TARGET_FILE_DIR:ortmt5>)
  endforeach()
endforeach()
