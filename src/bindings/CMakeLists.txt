foreach(ARCH IN LISTS MICROARCHS)
    pybind11_add_module(_backend_${ARCH} NO_EXTRAS 
        module.cpp
    )

    target_compile_definitions(_backend_${ARCH} PRIVATE BACKEND_NAME=_backend_${ARCH})
    target_link_libraries(_backend_${ARCH} PRIVATE engine_core_${ARCH})
    install(TARGETS _backend_${ARCH} DESTINATION affineflow_ml)
endforeach()