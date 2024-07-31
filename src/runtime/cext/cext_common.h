/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file cext_common.h
 * \brief Common utilities for c extension
 */
#ifndef TVM_RUNTIME_CEXT_CEXT_COMMON_H_
#define TVM_RUNTIME_CEXT_CEXT_COMMON_H_

#include <tvm/runtime/packed_func.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <string>
#include <vector>
#include <any>

#include <cstdarg>
#include <dlfcn.h>


namespace tvm {
namespace runtime {

typedef void (*bridge_call_t) (void**);

enum class CEXTResult {
    CEXT_SUCCESS = 0,
    CEXT_OPEN_LIB_FAIL = 1,
    CEXT_LOAD_FUNC_SYMBOL_FAIL = 2
};

std::string CompileStringCode(const std::string& code, const std::string& name, const std::string& dir) {
    // check dir path exist, if not create dir path
    if (!std::filesystem::exists(dir)) {
        try {
            std::filesystem::create_directories(dir);
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Failed to create directory: " << e.what() << std::endl;
        }
    }
    // create c source file path
    std::filesystem::path dir_path = dir;
    std::filesystem::path c_file_path = dir_path / (name + ".c");

    // save the string code
    std::ofstream csrc_file(c_file_path.string(), std::ios::out);
    if (!csrc_file) {
        throw std::runtime_error("Failed to create c source file: " + c_file_path.string());
    }
    csrc_file << code;
    csrc_file.close();

    // create shared lib path
    std::filesystem::path so_file_path = dir_path / ("lib" + name + ".so");
    std::string command = "gcc -std=c11 -o3 -fopenmp -pthread -fPIC -shared -o " + so_file_path.string() + " " + c_file_path.string();

    int ret = system(command.c_str());
    if (ret != 0) {
        std::cout << "Compilation failed: " << command << "\n";
        std::abort(); 
        throw std::runtime_error("Compilation failed. " + command);
    }

    return so_file_path.string();
}

std::string GenBridgeCode(int num_args, const std::string& name, const std::vector<DLDataType>& arg_types) {
    std::ostringstream oss;
    oss << "\nvoid bridge_call(void** void_args) {\n";
    // gen convert code first
    for (int i = 0; i < num_args; i++) {
        DLDataType t = arg_types[i];
        std::string arg_name = "arg_" + std::to_string(i);
        std::string local_str = "";
        if (t.code == kDLInt) {
            if (t.bits == 64U) {
                local_str = "int64_t " + arg_name + " = " + "*(int64_t*)" + "void_args[" + std::to_string(i) + "];";
            } else if (t.bits == 32U) {
                local_str = "int32_t " + arg_name + " = " + "*(int32_t*)" + "void_args[" + std::to_string(i) + "];";
            }
        } else if (t.code == kDLUInt) {
            if (t.bits == 32U) {
                local_str = "uint32_t " + arg_name + " = " + "*(uint32_t*)" + "void_args[" + std::to_string(i) + "];";
            }
        } else if (t.code == kDLFloat) {
            if (t.bits == 64U) {
                local_str = "float64_t " + arg_name + " = " + "*(float64_t*)" + "void_args[" + std::to_string(i) + "];";
            } else if (t.bits == 32U) {
                local_str = "float32_t " + arg_name + " = " + "*(float32_t*)" + "void_args[" + std::to_string(i) + "];";
            }
        } else if (t.code == kTVMOpaqueHandle) {
            local_str = "void* " + arg_name + " = " + "void_args[" + std::to_string(i) + "];";
        }
        local_str = "    " + local_str;
        // std::cout << "$$$$$$$$$ debug >>>> " << i << ", " << local_str << "\n";
        oss << local_str << "\n";
    }

    oss << "    " + name + "(";
    for (int i = 0; i < num_args - 1; i++) {
        oss << "arg_" + std::to_string(i) + ", ";
    }
    oss << "arg_" + std::to_string(num_args-1);
    oss << ");\n";
    oss << "}\n";
    return oss.str();
}

std::string GetHeaderCode() {
    // pad header files include 
    std::ostringstream oss;
    oss << "#include <stdlib.h>\n";
    oss << "#include <stdio.h>\n";
    oss << "#include <math.h>\n";
    oss << "#include <stdbool.h>\n";
    oss << "#include <stdint.h>\n\n";
    std::string pre_header = oss.str();

    return pre_header;
}

CEXTResult cextLaunchKernel(void** void_args, const std::string& kernel_name, const std::string& lib_name) {
    // open shared library
    void* handle = dlopen(lib_name.c_str(), RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error loading object file: " << dlerror() << std::endl;
        return CEXTResult::CEXT_OPEN_LIB_FAIL;
    }

    // load kernel function symbol    
    bridge_call_t bridge_call_func = nullptr;
    void* sym = dlsym(handle, "bridge_call");
    if (sym != nullptr) {
        bridge_call_func = reinterpret_cast<bridge_call_t>(sym);
    }
    if (bridge_call_func == nullptr) {
        std::cerr << "Can't find function: bridge_call " << "\n";
        std::abort();
        return CEXTResult::CEXT_LOAD_FUNC_SYMBOL_FAIL;
    }
    // std::cout << "################ bridge call: " << bridge_call_func << "\n";

    // utilize bridge_call func to call kernel func
    bridge_call_func(void_args);

    // std::cout << "@@@@@???? bridge call finish\n";

    // close dynamic library
    dlclose(handle);

    return CEXTResult::CEXT_SUCCESS;
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_CEXT_CEXT_COMMON_H_
