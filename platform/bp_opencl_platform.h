#pragma once

#include <iostream>
#include <vector>
#include <gsl/pointers>

#include "CL/opencl.h"

namespace platform {
class bp_platform {
public:
    bp_platform() : m_platforms{} {}
    bp_platform(const bp_platform&) = delete;
    bp_platform& operator=(const bp_platform&) = delete;
    bp_platform(bp_platform&&) = delete;
    bp_platform& operator=(bp_platform&&) = delete;

    void init();

    size_t get_number() const
    {
        return m_platforms.size();
    }

    cl_platform_id get_ith(size_t index) const
    {
        if (index >= m_platforms.size()) {
            std::cout << "Error: index out of range." << std::endl;
            return nullptr;
        }
        return m_platforms[index];
    }

    void print_info(gsl::not_null<cl_platform_id>) const;
private:
    std::vector<cl_platform_id> m_platforms;
};

class bp_device {
public:
    bp_device() : m_devices{} {}
    bp_device(const bp_device&) = delete;
    bp_device& operator=(const bp_device&) = delete;
    bp_device(bp_device&&) = delete;
    bp_device& operator=(bp_device&&) = delete;

    void init(gsl::not_null<cl_platform_id>);

    size_t get_number() const
    {
        return m_devices.size();
    }

    cl_device_id get_ith(size_t index) const
    {
        if (index >= m_devices.size()) {
            std::cout << "Error: device index out of range." << std::endl;
            return nullptr;
        }
        return m_devices[index];
    }

    void print_info(gsl::not_null<cl_device_id>) const;
private:
    std::vector<cl_device_id> m_devices;
};

class bp_context {
public:
    bp_context() : m_context{ nullptr } {}
    bp_context(const bp_context&) = delete;
    bp_context& operator=(const bp_context&) = delete;
    bp_context(bp_context&&) = delete;
    bp_context& operator=(bp_context&&) = delete;

    void init(gsl::not_null<cl_platform_id>, bp_device&);

    cl_context get() const
    {
        return m_context;
    }
private:
    cl_context m_context;
};
}
