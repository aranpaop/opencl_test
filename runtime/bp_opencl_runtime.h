#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <gsl/pointers>

#include "CL/opencl.h"

namespace runtime {
class bp_cmdqueue {
public:
    bp_cmdqueue(const bp_cmdqueue&) = delete;
    bp_cmdqueue& operator=(const bp_cmdqueue&) = delete;
    bp_cmdqueue(bp_cmdqueue&&) = delete;
    bp_cmdqueue& operator=(bp_cmdqueue&&) = delete;

    cl_command_queue create_command_queue(gsl::not_null<cl_context>, gsl::not_null<cl_device_id>);
};
}
