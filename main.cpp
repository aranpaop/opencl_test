#include <iostream>

#include "platform/bp_opencl_platform.h"
#include "runtime/bp_opencl_runtime.h"

int main()
{
    // Get all platforms.
    platform::bp_platform bp_platform{};
    bp_platform.init();
    size_t num_platform = bp_platform.get_number();
    for (auto i = 0; i < num_platform; ++i) {
        std::cout << "Info: platform " << i << std::endl;
        cl_platform_id platform = bp_platform.get_ith(i);
        bp_platform.print_info(platform);

        // Get all devices on this platform.
        platform::bp_device bp_device{};
        bp_device.init(platform);

        // Use platform and devices to create context.
        platform::bp_context bp_context{};
        bp_context.init(platform, bp_device);

        size_t num_device = bp_device.get_number();
        for (auto j = 0; j < num_device; ++j) {
            std::cout << "Info: device " << j << std::endl;
            cl_device_id device = bp_device.get_ith(i);
            bp_device.print_info(device);

            // Create command queue on this device.
            runtime::bp_cmdqueue bp_cmdqueue{};
            cl_context context = bp_context.get();
            cl_command_queue command_queue = bp_cmdqueue.create_command_queue(context, device);
        }
    }
    return 0;
}