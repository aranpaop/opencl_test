#pragma once

#include <iostream>
#include <vector>
#include <gsl/pointers>

#include "CL/opencl.h"

#include "../utils/bp_opencl_common.h"

namespace platform {
class bp_platform {
public:
    bp_platform();
    bp_platform(const bp_platform&) = delete;
    bp_platform& operator=(const bp_platform&) = delete;
    bp_platform(bp_platform&&) = delete;
    bp_platform& operator=(bp_platform&&) = delete;

    size_t get_number() const
    {
        return m_platforms.size();
    }

    cl_platform_id get_ith(size_t index) const
    {
        bp_validate_condition(index < m_platforms.size(), "Platform index out of range.");
        return m_platforms[index];
    }

    void print_info(gsl::not_null<cl_platform_id>) const;
private:
    std::vector<cl_platform_id> m_platforms;
};

class bp_device {
public:
    bp_device(gsl::not_null<cl_platform_id>);
    ~bp_device()
    {
        for (auto device : m_devices) {
            cl_int err = clReleaseDevice(device);
            bp_validate_condition(err == CL_SUCCESS, "Release devices failed.");
        }
    }
    bp_device(const bp_device&) = delete;
    bp_device& operator=(const bp_device&) = delete;
    bp_device(bp_device&&) = delete;
    bp_device& operator=(bp_device&&) = delete;

    size_t get_number() const
    {
        return m_devices.size();
    }

    cl_device_id get_ith(size_t index) const
    {
        bp_validate_condition(index < m_devices.size(), "Device index out of range.");
        return m_devices[index];
    }

    void print_info(gsl::not_null<cl_device_id>) const;
private:
    std::vector<cl_device_id> m_devices;
};

class bp_context {
public:
    bp_context(gsl::not_null<cl_platform_id>, bp_device&);
    ~bp_context()
    {
        cl_int err = clReleaseContext(m_context);
        bp_validate_condition(err == CL_SUCCESS, "Release context failed.");
    }
    bp_context(const bp_context&) = delete;
    bp_context& operator=(const bp_context&) = delete;
    bp_context(bp_context&&) = delete;
    bp_context& operator=(bp_context&&) = delete;

    cl_context get() const
    {
        return m_context;
    }
private:
    cl_context m_context;
};
}
