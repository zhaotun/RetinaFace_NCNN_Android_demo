cmake_minimum_required(VERSION 3.4.1)

#include头文件目录
include_directories(src/main/cpp/include
                    src/main/cpp/)

#source directory源文件目录
file(GLOB RETINAFACE_SRC src/main/cpp/*.h
                    src/main/cpp/*.cpp)
set(RETINAFACE_COMPILE_CODE ${RETINAFACE_SRC})


#添加ncnn库
add_library(libncnn STATIC IMPORTED )
set_target_properties(libncnn
  PROPERTIES IMPORTED_LOCATION
  ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libncnn.a)

#编译为动态库
add_library(retinaface SHARED ${RETINAFACE_COMPILE_CODE})

#添加工程所依赖的库
find_library(  log-lib log )
target_link_libraries( retinaface
                       libncnn
                       jnigraphics
                       z
                       ${log-lib} )
