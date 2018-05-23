cmake_minimum_required(VERSION 3.8)

project(shelfdetect)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


find_package( OpenCV REQUIRED )


set(ACTIVEPROJECT CVProject/code)

file(GLOB SOURCE_FILES
        "${ACTIVEPROJECT}/**.hpp"
        "${ACTIVEPROJECT}/**.cpp"
        "${ACTIVEPROJECT}/**.h")


add_executable(${PROJECT_NAME} ${SOURCE_FILES})
include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

