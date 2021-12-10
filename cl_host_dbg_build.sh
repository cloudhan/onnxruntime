#!/bin/bash

set -ev

./build.sh \
    --cmake_generator Ninja \
    --config Debug \
    --cmake_extra_defines CMAKE_EXPORT_COMPILE_COMMANDS=ON \
    --use_opencl \
    --use_nnapi \
    --minimal_build extended \
    --build_wheel \
    --build_java \
    --skip_submodule_sync --skip_tests \
