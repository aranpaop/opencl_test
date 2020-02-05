#include "bp_opencl_runtime.h"

#include <iostream>
#include <vector>
#include <string>
#include <gsl/pointers>

#include "CL/opencl.h"

#include "../utils/bp_opencl_common.h"
#include "../platform/bp_opencl_platform.h"

static void print_event_profiling_info(gsl::not_null<cl_event> event)
{
    cl_ulong queued_time, submit_time, start_time, end_time;
    cl_int err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queued_time, nullptr);
    bp_validate_condition(err == CL_SUCCESS, "Get event profiling info failed.");
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &submit_time, nullptr);
    bp_validate_condition(err == CL_SUCCESS, "Get event profiling info failed.");
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, nullptr);
    bp_validate_condition(err == CL_SUCCESS, "Get event profiling info failed.");
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, nullptr);
    bp_validate_condition(err == CL_SUCCESS, "Get event profiling info failed.");

    bp_print_info(true, "Time between queued and submit: ", submit_time - queued_time);
    bp_print_info(true, "Time between submit and start: ", start_time - submit_time);
    bp_print_info(true, "Time between start and end: ", end_time - start_time);
}

namespace runtime {
cl_command_queue bp_cmdqueue::create_command_queue(gsl::not_null<cl_context> context, gsl::not_null<cl_device_id> device)
{
    cl_int err;
    cl_command_queue command_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    bp_validate_condition(err == CL_SUCCESS, "Create command queue failed.");
    bp_print_info(true, "Successfully create command queue.");
    m_command_queues.push_back(command_queue);
    return command_queue;
}

void bp_cmdqueue::enqueue_kernel(gsl::not_null<cl_command_queue> command_queue, gsl::not_null<cl_kernel> kernel) const
{
    size_t global_work_size_x = TEST_GLOBAL_SIZE_X;
    cl_event event = nullptr;
    cl_int err = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, &global_work_size_x, nullptr, 0, nullptr, &event);
    bp_validate_condition(err == CL_SUCCESS, "Enqueue kernel failed.");
    bp_print_info(true, "Successfully enqueue kernel.");

    err = clWaitForEvents(1, &event);
    bp_validate_condition(err == CL_SUCCESS, "Wait for event failed.");

    print_event_profiling_info(event);

    err = clReleaseEvent(event);
    bp_validate_condition(err == CL_SUCCESS, "Release event failed.");
}

cl_program bp_program::create_program_with_source(gsl::not_null<cl_context> context,
    const std::vector<std::string>& kernel_funcs, platform::bp_device& bp_device)
{
    cl_uint count = kernel_funcs.size();
    auto strings(std::make_unique<const char*[]>(count));
    bp_print_info(true, "Kernel functions:");
    for (auto i = 0; i < count; ++i) {
        strings[i] = kernel_funcs[i].c_str();
        bp_print_info(true, "");
        bp_print_info(false, kernel_funcs[i]);
    }

    cl_int err;
    cl_program program = clCreateProgramWithSource(context, count, strings.get(), nullptr, &err);
    bp_validate_condition(err == CL_SUCCESS, "Create program failed.");
    bp_print_info(true, "Successfully create program.");

    cl_uint num_devices = bp_device.get_number();
    auto devices(std::make_unique<cl_device_id[]>(num_devices));
    for (auto i = 0; i < num_devices; ++i) {
        devices[i] = bp_device.get_ith(i);
    }
    err = clBuildProgram(program, num_devices, devices.get(), nullptr, nullptr, nullptr);
    bp_validate_condition(err == CL_SUCCESS, "Build program failed.");
    bp_print_info(true, "Successfully build program.");

    m_programs.push_back(program);

    return program;
}

cl_kernel bp_kernel::create_kernel(gsl::not_null<cl_program> program, const std::string& kernel_name)
{
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
    bp_validate_condition(err == CL_SUCCESS, "Create kernel failed.");
    bp_print_info(true, "Successfully create kernel.");

    m_kernels.push_back(kernel);

    return kernel;
}
}
