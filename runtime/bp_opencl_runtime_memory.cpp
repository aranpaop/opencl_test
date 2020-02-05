#include "bp_opencl_runtime_memory.h"

#include <vector>
#include <gsl/pointers>

#include "CL/opencl.h"

#include "../utils/bp_opencl_common.h"

namespace runtime {
namespace memory {
cl_mem bp_memory::create_buffer(gsl::not_null<cl_context> context, cl_mem_flags flags, size_t size, void* host_ptr)
{
    cl_int err;
    cl_mem buffer = clCreateBuffer(context, flags, size, host_ptr, &err);
    bp_validate_condition(err == CL_SUCCESS, "Create buffer failed.");
    bp_print_info(true, "Successfully create buffer.");

    m_memories.push_back(buffer);

    return buffer;
}
}
}