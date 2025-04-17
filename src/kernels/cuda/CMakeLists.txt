set(CU_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/1_naive.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/2_shared_mem_cache_blocking.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/3_1d_blocktiling.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/4_2d_blocktiling.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/5_vectorise_smem.cu
  PARENT_SCOPE
)