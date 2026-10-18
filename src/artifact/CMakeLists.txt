add_library(ninfer_artifact STATIC
  schema.cpp
  formats.cpp
  layouts.cpp
  file_io.cpp
  reader.cpp
  binder.cpp
  materializer.cpp
  transcode.cpp
  permute_row_split.cu
  views.cpp
)
ninfer_internal_includes(ninfer_artifact)
# permute_row_split.cu is the one device translation unit outside core and ops; it retags to
# HIP with the rest of the tree under the hip backend and is a no-op on CUDA.
ninfer_hipify_cu(ninfer_artifact)
target_link_libraries(ninfer_artifact PUBLIC ninfer_core ninfer::json)
