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