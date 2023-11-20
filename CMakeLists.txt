cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
# set(CMAKE_C_COMPILER /usr/bin/clang)

project(tglang VERSION 1.0.0 LANGUAGES C)
set(CMAKE_C_STANDARD 11)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_C_FLAGS_INIT                "-Wall -std=c11")
set(CMAKE_C_FLAGS_DEBUG_INIT          "-g")
set(CMAKE_C_FLAGS_MINSIZEREL_INIT     "-Os -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE_INIT        "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO_INIT "-O2 -g")

option(EMBED_WEIGHTS "If ON, then models' weights will be embed into binaries" OFF)
option(USE_AVX_EXP "If ON, uses experimental exponentials from avx_mathfun" ON)

if(EMBED_WEIGHTS)
    add_definitions(-DEMBED_WEIGHTS)
endif()

if(USE_AVX_EXP)
    add_definitions(-DUSE_AVX_EXP)
endif()

include_directories(src/dependencies)

# tokenizer, gru and svc can compile for a long time (if embedding is on)
# so we compile object files for them once and then use them multiple times
add_library(tokenizer_obj OBJECT src/tokenizer.c)
add_library(gru_obj OBJECT src/gru.c)
add_library(svc_obj OBJECT src/svc.c)

add_executable(test src/test.c
                    src/matrix.c
                    src/tglang.c
                    $<TARGET_OBJECTS:tokenizer_obj>
                    $<TARGET_OBJECTS:gru_obj>
                    $<TARGET_OBJECTS:svc_obj>)

add_executable(embed_weights src/embed_weights.c
                             src/matrix.c
                             $<TARGET_OBJECTS:tokenizer_obj>
                             $<TARGET_OBJECTS:gru_obj>
                             $<TARGET_OBJECTS:svc_obj>)

add_library(tokenizer SHARED $<TARGET_OBJECTS:tokenizer_obj>)
add_library(gru SHARED $<TARGET_OBJECTS:gru_obj> src/matrix.c)
add_library(tglang SHARED src/tglang.c
                          src/matrix.c
                          $<TARGET_OBJECTS:tokenizer_obj>
                          $<TARGET_OBJECTS:gru_obj>
                          $<TARGET_OBJECTS:svc_obj>)


# link math.h lib
target_link_libraries(test m pthread)
target_link_libraries(embed_weights m pthread)
target_link_libraries(gru m pthread)
target_link_libraries(tglang m pthread)

target_compile_options(test PUBLIC "-mavx2" "-ffast-math")
target_compile_options(embed_weights PUBLIC "-mavx2")
target_compile_options(tglang PUBLIC "-fPIC" "-mavx2" "-ffast-math")
target_compile_options(tokenizer_obj PUBLIC "-fPIC" "-mavx2")
target_compile_options(gru PUBLIC "-fPIC" "-mavx2" "-ffast-math")
target_compile_options(gru_obj PUBLIC "-fPIC" "-mavx2" "-ffast-math")
target_compile_options(svc_obj PUBLIC "-fPIC" "-mavx2" "-ffast-math")
