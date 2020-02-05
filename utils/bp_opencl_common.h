#pragma once

#include <iostream>

constexpr size_t MAX_STRING_LENGTH = 1024;

constexpr size_t TEST_GLOBAL_SIZE_X = 256;

inline void bp_validate_condition(bool condition, const std::string& message)
{
    if (!condition) {
        std::cout << condition << std::endl;
        std::cout << "Error: " << message << std::endl;
        exit(EXIT_FAILURE);
    }
}

template<typename T, typename ... Types>
inline void bp_print_info(bool print_head, const T& message, const Types& ... messages)
{
    if (print_head) {
        std::cout << "Info: ";
    }
    std::cout << message;
    bp_print_info(false, messages ...);
}

template<typename T>
inline void bp_print_info(bool print_head, const T& message)
{
    if (print_head) {
        std::cout << "Info: ";
    }
    std::cout << message << std::endl;
}

template<typename T>
void fill_random_data(T* ptr, size_t size, T top, T bottom)
{
    srand(0);
    for (auto i = 0; i < size; ++i) {
        ptr[i] = bottom + (top - bottom) * static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
    }
}
