#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <gsl/pointers>

#include "CL/opencl.h"

#include "../utils/bp_opencl_common.h"
#include "../platform/bp_opencl_platform.h"

namespace runtime {
class bp_cmdqueue {
public:
    bp_cmdqueue() : m_command_queues{} {}
    ~bp_cmdqueue()
    {
        for (auto command_queue : m_command_queues) {
            cl_int err = clReleaseCommandQueue(command_queue);
            bp_validate_condition(err == CL_SUCCESS, "Release command queue failed.");
        }
    }
    bp_cmdqueue(const bp_cmdqueue&) = delete;
    bp_cmdqueue& operator=(const bp_cmdqueue&) = delete;
    bp_cmdqueue(bp_cmdqueue&&) = delete;
    bp_cmdqueue& operator=(bp_cmdqueue&&) = delete;

    cl_command_queue create_command_queue(gsl::not_null<cl_context>, gsl::not_null<cl_device_id>);

    void enqueue_kernel(gsl::not_null<cl_command_queue>, gsl::not_null<cl_kernel>) const;
private:
    std::vector<cl_command_queue> m_command_queues;
};

class bp_program {
public:
    bp_program() : m_programs{} {}
    ~bp_program()
    {
        for (auto program : m_programs) {
            cl_int err = clReleaseProgram(program);
            bp_validate_condition(err == CL_SUCCESS, "Release platform failed.");
        }
    }
    bp_program(const bp_program&) = delete;
    bp_program& operator=(const bp_program&) = delete;
    bp_program(bp_program&&) = delete;
    bp_program& operator=(bp_program&&) = delete;

    cl_program create_program_with_source(gsl::not_null<cl_context>,
        const std::vector<std::string>&, platform::bp_device&);
private:
    std::vector<cl_program> m_programs;
};

class bp_kernel {
public:
    bp_kernel() : m_kernels{} {}
    ~bp_kernel()
    {
        for (auto kernel : m_kernels) {
            cl_int err = clReleaseKernel(kernel);
            bp_validate_condition(err == CL_SUCCESS, "Release kernel failed.");
        }
    }
    bp_kernel(const bp_kernel&) = delete;
    bp_kernel& operator=(const bp_kernel&) = delete;
    bp_kernel(bp_kernel&&) = delete;
    bp_kernel& operator=(bp_kernel&&) = delete;

    cl_kernel create_kernel(gsl::not_null<cl_program>, const std::string&);
private:
    std::vector<cl_kernel> m_kernels;
};

template<typename T, typename ... Types>
inline void set_args(gsl::not_null<cl_kernel> kernel, size_t arg_index, T& arg, Types& ... args)
{
    cl_int err = clSetKernelArg(kernel, arg_index, arg.second, arg.first);
    bp_validate_condition(err == CL_SUCCESS, "Set kernel arg failed.");
    bp_print_info(true, "Successfully set kernel arg ", arg_index);
    ++arg_index;
    set_args(kernel, arg_index, args ...);
}

template<typename T>
inline void set_args(gsl::not_null<cl_kernel> kernel, size_t arg_index, T& arg)
{
    cl_int err = clSetKernelArg(kernel, arg_index, arg.second, arg.first);
    bp_validate_condition(err == CL_SUCCESS, "Set kernel arg failed.");
    bp_print_info(true, "Successfully set kernel arg ", arg_index);
    bp_print_info(true, "All kernel args have been set.");
}
}
