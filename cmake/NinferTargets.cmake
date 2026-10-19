# Internal headers are private to each compile owner. External headers come from
# the dependency targets (or a component-local include for bundled C sources).
function(ninfer_internal_includes target)
  if(NINFER_BACKEND STREQUAL "hip")
    # The shadow headers go first: they are the <cuda*.h>, <cublas_v2.h> and <cub/...> the
    # sources include (src/hipshim/README.md). The CUDA build never looks at that directory.
    target_include_directories(${target} PRIVATE
      ${PROJECT_SOURCE_DIR}/src/hipshim
      ${PROJECT_SOURCE_DIR}/include
      ${PROJECT_SOURCE_DIR}/src
      ${NINFER_HIPCUB_INCLUDE_DIR})
  else()
    target_include_directories(${target} PRIVATE
      ${PROJECT_SOURCE_DIR}/include
      ${PROJECT_SOURCE_DIR}/src)
  endif()
endfunction()

# Under the HIP backend the .cu sources are compiled by hipcc instead of nvcc. CMake only maps
# .cu to the CUDA language, so each target retags its own sources here (the language property is
# directory-scoped and must be set where the target is defined). The file extension stays .cu so
# the CUDA build and every path-based filter keep working unchanged.
function(ninfer_hipify_cu target)
  if(NOT NINFER_BACKEND STREQUAL "hip")
    return()
  endif()
  get_target_property(_ninfer_hip_sources ${target} SOURCES)
  set(_ninfer_hip_cu_files "")
  foreach(_ninfer_src IN LISTS _ninfer_hip_sources)
    if(_ninfer_src MATCHES "\\.cu$")
      list(APPEND _ninfer_hip_cu_files ${_ninfer_src})
    endif()
  endforeach()
  if(_ninfer_hip_cu_files)
    set_source_files_properties(${_ninfer_hip_cu_files} PROPERTIES LANGUAGE HIP)
  endif()
endfunction()

function(ninfer_cuda_archive target)
  if(NINFER_BACKEND STREQUAL "hip")
    # RDC/device-link and CUDA_RUNTIME_LIBRARY are nvcc-specific; hipcc compiles whole programs
    # and the runtime comes from the hip:: link interface.
    ninfer_hipify_cu(${target})
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-lineinfo>)
    return()
  endif()
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ${NINFER_CUDA_RDC}
    CUDA_RESOLVE_DEVICE_SYMBOLS ${NINFER_CUDA_RDC})
  if(WIN32)
    set_target_properties(${target} PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  endif()
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endfunction()

function(ninfer_cuda_non_rdc_archive target)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
  if(WIN32)
    set_target_properties(${target} PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  endif()
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endfunction()
