add_executable(gemm main.cpp ${CPP_SOURCES})

target_include_directories(gemm
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
  PUBLIC
    ${PROJECT_BINARY_DIR}  # so consumers can see Config.h if needed
)

target_link_libraries(gemm
  PRIVATE
    gemm_with_openmp
)

if(NVIDIA)
  target_link_libraries(gemm
  PRIVATE
    cuda_kernels
)

elseif(AMD)
  # target_link_libraries(${PROJECT_NAME}
  #   roc::rocthrust
  #   roc::hipsparse
  # )
elseif(CPU)
endif()

add_custom_target(run_gemm
  COMMAND ${CMAKE_COMMAND} -E echo "🚀 Running GEMM..."
  COMMAND ./gemm
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/examples/gemm
  DEPENDS gemm
)