cmake_minimum_required(VERSION 3.8)

project(ProjectWork)


find_package( OpenCV REQUIRED )
find_package( Boost 1.66.0 COMPONENTS filesystem REQUIRED)

set(CMAKE_CXX_STANDARD 14)

set(ACTIVEPROJECT CVProject/code)

file(GLOB SOURCE_FILES
        "${ACTIVEPROJECT}/**.hpp"
        "${ACTIVEPROJECT}/**.cpp"
        "${ACTIVEPROJECT}/**.h")


add_executable(${PROJECT_NAME} ${SOURCE_FILES})
include_directories(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})


target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES})

