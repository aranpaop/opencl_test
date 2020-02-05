#include "bp_opencl_platform.h"

#include <iostream>
#include <vector>
#include <string>
#include <gsl/pointers>

#include "CL/opencl.h"

#include "../utils/bp_opencl_common.h"

static inline void print_platform_info(gsl::not_null<cl_platform_id> platform, cl_platform_info info)
{
    auto info_string(std::make_unique<char[]>(MAX_STRING_LENGTH));
    cl_int err = clGetPlatformInfo(platform, info, MAX_STRING_LENGTH, info_string.get(), nullptr);
    bp_validate_condition(err == CL_SUCCESS, "Get platform info failed.");
    std::string platform_info{ info_string.get() };
    bp_print_info(false, platform_info);
}

static inline void print_device_info(gsl::not_null<cl_device_id> device, cl_device_info info)
{
    auto info_string(std::make_unique<char[]>(MAX_STRING_LENGTH));
    cl_int err = clGetDeviceInfo(device, info, MAX_STRING_LENGTH, info_string.get(), nullptr);
    bp_validate_condition(err == CL_SUCCESS, "Get device info failed.");
    std::string device_info{ info_string.get() };
    bp_print_info(false, device_info);
}

template<typename T>
static inline T get_device_info_single_type(gsl::not_null<cl_device_id> device, cl_device_info info)
{
    T info_type{};
    cl_int err = clGetDeviceInfo(device, info, sizeof(T), &info_type, nullptr);
    bp_validate_condition(err == CL_SUCCESS, "Get device info failed.");
    return info_type;
}

namespace platform {
bp_platform::bp_platform()
{
    cl_uint num_platform;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platform);
    bp_validate_condition(err == CL_SUCCESS, "Get platform number failed.");

    auto platforms(std::make_unique<cl_platform_id[]>(num_platform));
    err = clGetPlatformIDs(num_platform, platforms.get(), nullptr);
    bp_validate_condition(err == CL_SUCCESS, "Get platforms failed.");
    bp_print_info(true, "Successfully get platforms.");

    for (auto i = 0; i < num_platform; ++i) {
        m_platforms.push_back(platforms[i]);
    }
}

void bp_platform::print_info(gsl::not_null<cl_platform_id> platform) const
{
    std::cout << "Info: The name of platform is: ";
    print_platform_info(platform, CL_PLATFORM_NAME);
    std::cout << "Info: The vendor of platform is: ";
    print_platform_info(platform, CL_PLATFORM_VENDOR);
    std::cout << "Info: The version of platform is: ";
    print_platform_info(platform, CL_PLATFORM_VERSION);
    std::cout << "Info: The profile of platform is: ";
    print_platform_info(platform, CL_PLATFORM_PROFILE);
}

bp_device::bp_device(gsl::not_null<cl_platform_id> platform)
{
    cl_uint num_device;
    cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_device);
    bp_validate_condition(err == CL_SUCCESS, "Get device number failed.");

    auto devices(std::make_unique<cl_device_id[]>(num_device));
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_device, devices.get(), nullptr);
    bp_validate_condition(err == CL_SUCCESS, "Get devices failed.");
    bp_print_info(true, "Successfully get devices.");

    for (auto i = 0; i < num_device; ++i) {
        m_devices.push_back(devices[i]);
    }
}

void bp_device::print_info(gsl::not_null<cl_device_id> device) const
{
    std::cout << "Info: The name of device is: ";
    print_device_info(device, CL_DEVICE_NAME);
    std::cout << "Info: The vendor of device is: ";
    print_device_info(device, CL_DEVICE_VENDOR);
    std::cout << "Info: The version of device is: ";
    print_device_info(device, CL_DEVICE_VERSION);
    std::cout << "Info: The profile of device is: ";
    print_device_info(device, CL_DEVICE_PROFILE);
    std::cout << "Info: The version of driver is: ";
    print_device_info(device, CL_DRIVER_VERSION);

    cl_device_type device_type = get_device_info_single_type<cl_device_type>(device, CL_DEVICE_TYPE);
    std::cout << "Info: The type of device is: ";
    switch (device_type) {
        case CL_DEVICE_TYPE_CPU:
            bp_print_info(false, "CL_DEVICE_TYPE_CPU");
            break;
        case CL_DEVICE_TYPE_GPU:
            bp_print_info(false, "CL_DEVICE_TYPE_GPU");
            break;
        case CL_DEVICE_TYPE_ACCELERATOR:
            bp_print_info(false, "CL_DEVICE_TYPE_ACCELERATOR");
            break;
        case CL_DEVICE_TYPE_CUSTOM:
            bp_print_info(false, "CL_DEVICE_TYPE_CUSTOM");
            break;
        default:
            bp_print_info(false, "Unknown device type");
            break;
    }

    auto max_compute_units = get_device_info_single_type<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS);
    bp_print_info(true, "The max compute units of device is: ", max_compute_units);

    auto max_work_item_dimensions = get_device_info_single_type<cl_uint>(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
    bp_print_info(true, "The max work item dimensions of device is: ", max_work_item_dimensions);

    auto max_work_item_sizes(std::make_unique<size_t[]>(max_work_item_dimensions));
    cl_int err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * max_work_item_dimensions, max_work_item_sizes.get(), nullptr);
    bp_validate_condition(err == CL_SUCCESS, "Get device info failed.");
    std::cout << "Info: The max work item sizes of device are: (";
    for (auto i = 0; i < max_work_item_dimensions; ++i) {
        std::cout << max_work_item_sizes[i] << ", ";
    }
    bp_print_info(false, ")");

    auto max_work_group_size = get_device_info_single_type<size_t>(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
    bp_print_info(true, "The max work group size of device is: ", max_work_group_size);

    auto single_fp_config = get_device_info_single_type<cl_device_fp_config>(device, CL_DEVICE_SINGLE_FP_CONFIG);
    if ((single_fp_config & CL_FP_DENORM) == 0) {
        bp_print_info(true, "Device doesn't support denormal number of float.");
    } else {
        bp_print_info(true, "Device supports denormal number of float.");
    }

    auto double_fp_config = get_device_info_single_type<cl_device_fp_config>(device, CL_DEVICE_DOUBLE_FP_CONFIG);
    if ((double_fp_config & CL_FP_DENORM) == 0) {
        bp_print_info(true, "Device doesn't support denormal number of double.");
    } else {
        bp_print_info(true, "Device supports denormal number of double.");
    }
}

bp_context::bp_context(gsl::not_null<cl_platform_id> platform, bp_device& bp_device)
{
    const cl_context_properties prop[] = {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties>(platform.get()),
        0
    };

    auto devices(std::make_unique<cl_device_id[]>(bp_device.get_number()));
    for (auto i = 0; i < bp_device.get_number(); ++i) {
        devices[i] = bp_device.get_ith(i);
    }

    cl_int err;
    m_context = clCreateContext(prop, bp_device.get_number(), devices.get(), nullptr, nullptr, &err);
    bp_validate_condition(err == CL_SUCCESS, "Create context failed.");
    bp_print_info(true, "Successfully create context.");
}
}
