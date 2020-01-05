#include "bp_opencl_runtime.h"

#include <iostream>
#include <gsl/pointers>

#include "CL/opencl.h"

namespace runtime {
cl_command_queue bp_cmdqueue::create_command_queue(gsl::not_null<cl_context> context, gsl::not_null<cl_device_id> device)
{
    cl_int err;
    cl_command_queue command_queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error: create command queue failed." << std::endl;
        return nullptr;
    }
    std::cout << "Info: successfully create command queue." << std::endl;
    return command_queue;
}
}
