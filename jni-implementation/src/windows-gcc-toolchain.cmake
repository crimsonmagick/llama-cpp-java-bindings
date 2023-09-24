set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR "x86_64")
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

# comment out for posix threads
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mthreads")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mthreads")

set(USING_MINGW_TOOLCHAIN TRUE)
# set for posix threads
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mposix")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mposix")
