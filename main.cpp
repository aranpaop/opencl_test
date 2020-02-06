#include "platform/bp_opencl_platform.h"
#include "runtime/bp_opencl_runtime.h"
#include "runtime/bp_opencl_runtime_memory.h"
#include "utils/bp_opencl_common.h"

std::vector<std::string> kernel_funcs{
    "__kernel void float_func(__global const float* in, __global float* out)\n"
    "{\n"
    "\tint tid = get_global_id(0);\n"
    "\tout[tid] = in[tid * 3] * in[tid * 3 + 1] + in[tid * 3 + 2];\n"
    "}",
    "__kernel void double_func(__global const double* in, __global double* out)\n"
    "{\n"
    "\tint tid = get_global_id(0);\n"
    "\tout[tid] = in[tid * 3] * in[tid * 3 + 1] + in[tid * 3 + 2];\n"
    "}"
};

std::vector<std::string> kernel_names{
    "float_func",
    "double_func"
};

int main()
{
    // Get all platforms.
    platform::bp_platform bp_platform{};
    size_t num_platform = bp_platform.get_number();
    for (auto i = 0; i < num_platform; ++i) {
        bp_print_info(true, "Platform ", i);
        cl_platform_id platform = bp_platform.get_ith(i);
        bp_platform.print_info(platform);

        // Get all devices on this platform.
        platform::bp_device bp_device{ platform };

        // Use platform and devices to create context.
        platform::bp_context bp_context{ platform, bp_device };

        // Use context and devices to create program.
        runtime::bp_program bp_program{};
        cl_program program = bp_program.create_program_with_source(bp_context.get(), kernel_funcs, bp_device);

        // Create float and double test kernels.
        runtime::bp_kernel bp_kernel{};
        std::vector<cl_kernel> cl_kernels{};
        cl_kernel float_kernel = bp_kernel.create_kernel(program, kernel_names[0]);
        cl_kernel double_kernel = bp_kernel.create_kernel(program, kernel_names[1]);

        runtime::memory::bp_memory bp_memory{};
        cl_context context = bp_context.get();

        // Create float test objects and set to kernel.
        auto float_in(std::make_unique<float[]>(TEST_GLOBAL_SIZE_X * 3));
        auto float_out(std::make_unique<float[]>(TEST_GLOBAL_SIZE_X));
        fill_random_data<float>(float_in.get(), TEST_GLOBAL_SIZE_X * 3, 127.0f, -128.0f);
        cl_mem float_in_mem = bp_memory.create_buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            TEST_GLOBAL_SIZE_X * 3, float_in.get());
        cl_mem float_out_mem = bp_memory.create_buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
            TEST_GLOBAL_SIZE_X, float_out.get());
        runtime::set_args(float_kernel, 0, std::pair<cl_mem*, size_t>(&float_in_mem, sizeof(cl_mem)),
            std::pair<cl_mem*, size_t>(&float_out_mem, sizeof(cl_mem)));

        // Create float test objects and set to kernel.
        auto double_in(std::make_unique<double[]>(TEST_GLOBAL_SIZE_X * 3));
        auto double_out(std::make_unique<double[]>(TEST_GLOBAL_SIZE_X));
        fill_random_data<double>(double_in.get(), TEST_GLOBAL_SIZE_X * 3, 127.0, -128.0);
        cl_mem double_in_mem = bp_memory.create_buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            TEST_GLOBAL_SIZE_X * 3, double_in.get());
        cl_mem double_out_mem = bp_memory.create_buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
            TEST_GLOBAL_SIZE_X, double_out.get());
        runtime::set_args(double_kernel, 0, std::pair<cl_mem*, size_t>(&double_in_mem, sizeof(cl_mem)),
            std::pair<cl_mem*, size_t>(&double_out_mem, sizeof(cl_mem)));

        size_t num_device = bp_device.get_number();
        for (auto j = 0; j < num_device; ++j) {
            bp_print_info(true, "Device ", j);
            cl_device_id device = bp_device.get_ith(j);
            bp_device.print_info(device);

            // Create command queue on this device.
            runtime::bp_cmdqueue bp_cmdqueue{};
            cl_command_queue command_queue = bp_cmdqueue.create_command_queue(context, device);
            bp_cmdqueue.enqueue_kernel(command_queue, float_kernel);
            bp_cmdqueue.enqueue_kernel(command_queue, double_kernel);
        }
    }
    return 0;
}

// Test result:
/*
Info: Successfully get platforms.
Info: Platform 0
Info: The name of platform is: NVIDIA CUDA
Info: The vendor of platform is: NVIDIA Corporation
Info: The version of platform is: OpenCL 1.2 CUDA 10.1.236
Info: The profile of platform is: FULL_PROFILE
Info: Successfully get devices.
Info: Successfully create context.
Info: Kernel functions:
Info:
__kernel void float_func(__global const float* in, __global float* out)
{
        int tid = get_global_id(0);
        out[tid] = in[tid * 3] * in[tid * 3 + 1] + in[tid * 3 + 2];
}
Info:
__kernel void double_func(__global const double* in, __global double* out)
{
        int tid = get_global_id(0);
        out[tid] = in[tid * 3] * in[tid * 3 + 1] + in[tid * 3 + 2];
}
Info: Successfully create program.
Info: Successfully build program.
Info: Successfully create kernel.
Info: Successfully create kernel.
Info: Successfully create buffer.
Info: Successfully create buffer.
Info: Successfully set kernel arg 0
Info: Successfully set kernel arg 1
Info: All kernel args have been set.
Info: Successfully create buffer.
Info: Successfully create buffer.
Info: Successfully set kernel arg 0
Info: Successfully set kernel arg 1
Info: All kernel args have been set.
Info: Device 0
Info: The name of device is: GeForce MX250
Info: The vendor of device is: NVIDIA Corporation
Info: The version of device is: OpenCL 1.2 CUDA
Info: The profile of device is: FULL_PROFILE
Info: The version of driver is: 426.00
Info: The type of device is: CL_DEVICE_TYPE_GPU
Info: The max compute units of device is: 3
Info: The max work item dimensions of device is: 3
Info: The max work item sizes of device are: (1024, 1024, 64, )
Info: The max work group size of device is: 1024
Info: Device supports denormal number of float.
Info: Device supports denormal number of double.
Info: Successfully create command queue.
Info: Successfully enqueue kernel.
Info: Time between queued and submit: 116288
Info: Time between submit and start: 25376
Info: Time between start and end: 7264
Info: Successfully enqueue kernel.
Info: Time between queued and submit: 124704
Info: Time between submit and start: 25056
Info: Time between start and end: 5120
Info: Platform 1
Info: The name of platform is: Intel(R) OpenCL
Info: The vendor of platform is: Intel(R) Corporation
Info: The version of platform is: OpenCL 2.1
Info: The profile of platform is: FULL_PROFILE
Info: Successfully get devices.
Info: Successfully create context.
Info: Kernel functions:
Info:
__kernel void float_func(__global const float* in, __global float* out)
{
        int tid = get_global_id(0);
        out[tid] = in[tid * 3] * in[tid * 3 + 1] + in[tid * 3 + 2];
}
Info:
__kernel void double_func(__global const double* in, __global double* out)
{
        int tid = get_global_id(0);
        out[tid] = in[tid * 3] * in[tid * 3 + 1] + in[tid * 3 + 2];
}
Info: Successfully create program.
Info: Successfully build program.
Info: Successfully create kernel.
Info: Successfully create kernel.
Info: Successfully create buffer.
Info: Successfully create buffer.
Info: Successfully set kernel arg 0
Info: Successfully set kernel arg 1
Info: All kernel args have been set.
Info: Successfully create buffer.
Info: Successfully create buffer.
Info: Successfully set kernel arg 0
Info: Successfully set kernel arg 1
Info: All kernel args have been set.
Info: Device 0
Info: The name of device is: Intel(R) UHD Graphics 620
Info: The vendor of device is: Intel(R) Corporation
Info: The version of device is: OpenCL 2.1 NEO
Info: The profile of device is: FULL_PROFILE
Info: The version of driver is: 26.20.100.7262
Info: The type of device is: CL_DEVICE_TYPE_GPU
Info: The max compute units of device is: 24
Info: The max work item dimensions of device is: 3
Info: The max work item sizes of device are: (256, 256, 256, )
Info: The max work group size of device is: 256
Info: Device supports denormal number of float.
Info: Device supports denormal number of double.
Info: Successfully create command queue.
Info: Successfully enqueue kernel.
Info: Time between queued and submit: 213600
Info: Time between submit and start: 3905317
Info: Time between start and end: 6083
Info: Successfully enqueue kernel.
Info: Time between queued and submit: 26800
Info: Time between submit and start: 366700
Info: Time between start and end: 5333
Info: Device 1
Info: The name of device is: Intel(R) Core(TM) i7-8565U CPU @ 1.80GHz
Info: The vendor of device is: Intel(R) Corporation
Info: The version of device is: OpenCL 2.1 (Build 0)
Info: The profile of device is: FULL_PROFILE
Info: The version of driver is: 7.6.0.0814
Info: The type of device is: CL_DEVICE_TYPE_CPU
Info: The max compute units of device is: 8
Info: The max work item dimensions of device is: 3
Info: The max work item sizes of device are: (8192, 8192, 8192, )
Info: The max work group size of device is: 8192
Info: Device supports denormal number of float.
Info: Device supports denormal number of double.
Info: Successfully create command queue.
Info: Successfully enqueue kernel.
Info: Time between queued and submit: 1800
Info: Time between submit and start: 2127500
Info: Time between start and end: 251500
Info: Successfully enqueue kernel.
Info: Time between queued and submit: 1100
Info: Time between submit and start: 13100
Info: Time between start and end: 50300
*/