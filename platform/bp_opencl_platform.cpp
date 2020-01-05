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
    if (err != CL_SUCCESS) {
        std::cout << "Error: get platform info failed." << std::endl;
        return;
    }
    std::string platform_info{ info_string.get() };
    std::cout << platform_info << std::endl;
}

static inline void print_device_info(gsl::not_null<cl_device_id> device, cl_device_info info)
{
    auto info_string(std::make_unique<char[]>(MAX_STRING_LENGTH));
    cl_int err = clGetDeviceInfo(device, info, MAX_STRING_LENGTH, info_string.get(), nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Error: get device info failed." << std::endl;
        return;
    }
    std::string device_info{ info_string.get() };
    std::cout << device_info << std::endl;
}

template<typename T>
static inline T get_device_info_single_type(gsl::not_null<cl_device_id> device, cl_device_info info)
{
    T info_type{};
    cl_int err = clGetDeviceInfo(device, info, sizeof(T), &info_type, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Error: get device info failed." << std::endl;
    }
    return info_type;
}

namespace platform {
void bp_platform::init()
{
    cl_uint num_platform;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platform);
    if (err != CL_SUCCESS) {
        std::cout << "Error: get platform number failed." << std::endl;
        return;
    }
    auto platforms(std::make_unique<cl_platform_id[]>(num_platform));
    err = clGetPlatformIDs(num_platform, platforms.get(), nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Error: get platforms failed." << std::endl;
        return;
    }
    std::cout << "Info: successfully get platforms." << std::endl;
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

void bp_device::init(gsl::not_null<cl_platform_id> platform)
{
    cl_uint num_device;
    cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_device);
    if (err != CL_SUCCESS) {
        std::cout << "Error: get device number failed." << std::endl;
        return;
    }
    auto devices(std::make_unique<cl_device_id[]>(num_device));
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_device, devices.get(), nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Error: get devices failed." << std::endl;
        return;
    }
    std::cout << "Info: successfully get devices." << std::endl;
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
            std::cout << "CL_DEVICE_TYPE_CPU" << std::endl;
            break;
        case CL_DEVICE_TYPE_GPU:
            std::cout << "CL_DEVICE_TYPE_GPU" << std::endl;
            break;
        case CL_DEVICE_TYPE_ACCELERATOR:
            std::cout << "CL_DEVICE_TYPE_ACCELERATOR" << std::endl;
            break;
        case CL_DEVICE_TYPE_CUSTOM:
            std::cout << "CL_DEVICE_TYPE_CUSTOM" << std::endl;
            break;
        default:
            std::cout << "Unknown device type" << std::endl;
            break;
    }

    auto max_compute_units = get_device_info_single_type<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS);
    std::cout << "Info: The max compute units of device is: " << max_compute_units << std::endl;
    auto max_work_item_dimensions = get_device_info_single_type<cl_uint>(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
    std::cout << "Info: The max work item dimensions of device is: " << max_work_item_dimensions << std::endl;
    auto max_work_item_sizes(std::make_unique<size_t[]>(max_work_item_dimensions));
    cl_int err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * max_work_item_dimensions, max_work_item_sizes.get(), nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Error: get device info failed." << std::endl;
    }
    std::cout << "Info: The max work item sizes of device are: (";
    for (auto i = 0; i < max_work_item_dimensions; ++i) {
        std::cout << max_work_item_sizes.get()[i] << ", ";
    }
    std::cout << ")" << std::endl;
    auto max_work_group_size = get_device_info_single_type<size_t>(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
    std::cout << "Info: The max work group size of device is: " << max_work_group_size << std::endl;

    auto single_fp_config = get_device_info_single_type<cl_device_fp_config>(device, CL_DEVICE_SINGLE_FP_CONFIG);
    if ((single_fp_config & CL_FP_DENORM) == 0) {
        std::cout << "Info: Device doesn't support denormal number of float." << std::endl;
    } else {
        std::cout << "Info: Device supports denormal number of float." << std::endl;
    }
    auto double_fp_config = get_device_info_single_type<cl_device_fp_config>(device, CL_DEVICE_DOUBLE_FP_CONFIG);
    if ((double_fp_config & CL_FP_DENORM) == 0) {
        std::cout << "Info: Device doesn't support denormal number of double." << std::endl;
    } else {
        std::cout << "Info: Device supports denormal number of double." << std::endl;
    }
}

void bp_context::init(gsl::not_null<cl_platform_id> platform, bp_device& bp_device)
{
    const cl_context_properties prop[] = {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties>(platform.get()),
        0
    };
    auto devices(std::make_unique<cl_device_id[]>(bp_device.get_number()));
    for (auto i = 0; i < bp_device.get_number(); ++i) {
        devices.get()[i] = bp_device.get_ith(i);
    }
    cl_int err;
    m_context = clCreateContext(prop, bp_device.get_number(), devices.get(), nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cout << "Error: create context failed." << std::endl;
        return;
    }
    std::cout << "Info: successfully create context." << std::endl;
}
}
