# Project configuration
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(pytorch-cpp-ml-learning VERSION 1.0.0 LANGUAGES CXX)

# External Cmake files
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Set the pytorch version and find it
set(PYTORCH_VERSION "1.8.0")

find_package(Torch ${PYTORCH_VERSION} EXACT QUIET PATHS "${CMAKE_SOURCE_DIR}/libtorch")

if(NOT Torch_FOUND)
    unset(Torch_FOUND)
    include(fetch_libtorch)
endif()

# Project configuration
set(EXECUTABLE_NAME pytorch-cpp-ml-learning)

add_executable(${EXECUTABLE_NAME})
target_sources(${EXECUTABLE_NAME} PRIVATE main.cpp)

set_target_properties(${EXECUTABLE_NAME} PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)
target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES})

# Datasets
# Set and create the folder
set(DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data CACHE PATH "Dataset download directory")
file(MAKE_DIRECTORY ${DATA_DIR})

# MNIST
add_custom_target(
    mnist COMMAND ${CMAKE_COMMAND} -D DATA_DIR=${DATA_DIR} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/fetch_mnist.cmake
)

# Sub directories
add_subdirectory("src/mnist_classifier")
add_subdirectory("src/mnist_gan")
add_subdirectory("src/regression")

add_custom_target(src)

add_dependencies(src
    mnist_classifier
    mnist_gan
    regression
)

# Windows config
if(MSVC)
    include(copy_torch_dlls)
    copy_torch_dlls(${EXECUTABLE_NAME})
endif()
