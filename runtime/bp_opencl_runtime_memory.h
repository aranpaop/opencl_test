#pragma once

#include <vector>
#include <gsl/pointers>

#include "CL/opencl.h"

#include "../utils/bp_opencl_common.h"

namespace runtime {
namespace memory {
class bp_memory {
public:
    bp_memory() : m_memories{} {}
    ~bp_memory()
    {
        for (auto memory : m_memories) {
            cl_int err = clReleaseMemObject(memory);
            bp_validate_condition(err == CL_SUCCESS, "Release memory object failed.");
        }
    }
    bp_memory(const bp_memory&) = delete;
    bp_memory& operator=(const bp_memory&) = delete;
    bp_memory(bp_memory&&) = delete;
    bp_memory& operator=(bp_memory&&) = delete;

    cl_mem create_buffer(gsl::not_null<cl_context>, cl_mem_flags, size_t, void*);
private:
    std::vector<cl_mem> m_memories;
};
}
}
