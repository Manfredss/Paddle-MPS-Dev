add_subdirectory(allocation)

set(MEMORY_SRCS
  malloc.cc
  memcpy.cc
  stats.cc
  mem_utils.cc
  mem_visitor.cc)

collect_srcs(
  core_srcs
  SRCS
  ${MEMORY_SRCS})
