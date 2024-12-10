#
# unet

set(TEST_TARGET unet)
add_executable(${TEST_TARGET} unet.cpp unet-image.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common)
