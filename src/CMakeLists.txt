set(ENGINE_SOURCES
    ml/PCA.cpp
    ml/KNeighborsClassifier.cpp
)

# Determine architecture target capabilities
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64)")
    if(MSVC)
        set(MICROARCHS "generic" "avx2")
    else()
        set(MICROARCHS "generic" "avx2" "avx512")
    endif()
else()
    set(MICROARCHS "generic")
endif()

foreach(ARCH IN LISTS MICROARCHS)
    add_library(engine_core_${ARCH} OBJECT ${ENGINE_SOURCES})

    set_target_properties(engine_core_${ARCH} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(engine_core_${ARCH} PUBLIC ${CMAKE_SOURCE_DIR}/include)

    # Apply hardware optimization flags
    if(EMSCRIPTEN)
        target_compile_options(engine_core_${ARCH} PUBLIC "-msimd128")
    else()
        if(ARCH STREQUAL "avx2")
            if(MSVC)
                target_compile_options(engine_core_${ARCH} PUBLIC "/arch:AVX2")
            else()
                target_compile_options(engine_core_${ARCH} PUBLIC "-mavx2" "-mfma")
            endif()
        elseif(ARCH STREQUAL "avx512")
            if(MSVC)
                target_compile_options(engine_core_${ARCH} PUBLIC "/arch:AVX512")
            else()
                target_compile_options(engine_core_${ARCH} PUBLIC "-mavx512f" "-mavx512cd" "-mavx512vl" "-mavx512bw" "-mavx512dq" "-mfma")
            endif()
        endif()
    endif()

    # Link dependencies to the object target
    if(DEV_MODE)
        target_link_libraries(engine_core_${ARCH} PUBLIC Eigen3::Eigen)
    else()
        target_include_directories(engine_core_${ARCH} SYSTEM PUBLIC ${eigen_SOURCE_DIR})
    endif()

    if(APPLE)
        target_link_libraries(engine_core_${ARCH} PUBLIC "-framework Accelerate")
        target_compile_definitions(engine_core_${ARCH} PUBLIC EIGEN_USE_BLAS)
    elseif(NOT MSVC AND NOT EMSCRIPTEN)
        target_link_libraries(engine_core_${ARCH} PUBLIC BLAS::BLAS)
        target_compile_definitions(engine_core_${ARCH} PUBLIC EIGEN_USE_BLAS)
    endif()

    if(OpenMP_CXX_FOUND AND NOT EMSCRIPTEN)
        target_link_libraries(engine_core_${ARCH} PUBLIC OpenMP::OpenMP_CXX)
        if(MSVC)
            target_compile_options(engine_core_${ARCH} PUBLIC /openmp:experimental)
        endif()
    endif()
    
    target_compile_definitions(engine_core_${ARCH} PRIVATE EIGEN_USE_THREADS)
endforeach()

add_library(AffineFlow-ML::Core ALIAS engine_core_generic)
add_subdirectory(bindings)