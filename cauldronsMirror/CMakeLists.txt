cmake_minimum_required(VERSION 2.8.3)
project(cauldronsMirror)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(cauldronsMirror
    main.cpp
    src/GroggyDetector.cpp
)

target_link_libraries(cauldronsMirror
    ${OpenCV_LIBRARIES}
)