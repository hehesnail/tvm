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
 * \file cext_module.cc
 */
#include <tvm/runtime/registry.h>

#include <array>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../file_utils.h"
#include "../meta_data.h"
#include "../pack_args.h"
#include "cext_module.h"
#include "cext_common.h"


namespace tvm {
namespace runtime {

// Module to support c source file cpu execution
class CEXTModuleNode : public runtime::ModuleNode {
    public:
        explicit CEXTModuleNode(std::string code, std::string clean_code, std::string format,
                                std::unordered_map<std::string, FunctionInfo> fmap)
        : code_(code), clean_code_(clean_code), fmt_(format), fmap_(fmap) {}

        const char* type_key() const final { return "cext"; }

        int GetPropertyMask() const final {
            return ModulePropertyMask::kRunnable;
        }

        PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final;

        void SaveToFile(const String& file_name, const String& format) {
            std::string fmt = GetFileFormat(file_name, format);
            std::string meta_file = GetMetaFilePath(file_name);
            if (fmt == "c") {
                ICHECK_NE(code_.length(), 0);
                SaveMetaDataToFile(meta_file, fmap_);
                SaveBinaryToFile(file_name, code_);
            } else {
                LOG(FATAL) << "CEXTError: only support save c source file.";
            }
        }

        String GetSource(const String& format) final {
            if (format == fmt_) return code_;
            if (code_.length() != 0) {
                return code_;
            } else {
                return "";
            }
        }

        void SetSource(const String& code) final {
            code_ = code;
        }

        String GetPureSource(const String& format) final {
            if (clean_code_.length() != 0) {
                return clean_code_;
            } else {
                return "";
            }
        }

    private:
        // codegen c source code
        std::string code_;
        // codegen c clean source code
        std::string clean_code_;
        // format
        std::string fmt_;
        // function information table
        std::unordered_map<std::string, FunctionInfo> fmap_;
};


// a wrapped function class to get packed func
class CEXTWrappedFunc {
    public:
        // initialize the cext function
        void Init(CEXTModuleNode* m, ObjectPtr<Object> sptr, const std::string& func_name,
                  size_t num_void_args, const std::vector<DLDataType>& arg_types) {
            m_ = m;
            sptr_ = sptr;
            func_name_ = func_name;
            num_void_args_ = num_void_args;
            arg_types_ = arg_types;
        }

        // invoke the function with void arguments
        void operator()(TVMArgs args, TVMRetValue* rv, void** void_args) const {
            // get c device code, compile to dynamic lib
            std::string c_source_code = m_->GetSource("c");
            std::string c_header_code = GetHeaderCode();
            std::string c_bridge_code = GenBridgeCode(num_void_args_, func_name_, arg_types_);
            std::string c_merged_code = c_header_code + c_source_code + c_bridge_code;
            std::string lib_name = CompileStringCode(c_merged_code, func_name_, "./tmp/");
            // std::cout << "@@@@@@@@@@----> " << lib_name << "\n";

            // launch c kernel
            CEXTResult result = cextLaunchKernel(void_args, func_name_, lib_name);
            if (result != CEXTResult::CEXT_SUCCESS) {
                std::ostringstream os;
                os << "CEXTLaunch Error: " << static_cast<int>(result) << "\n"
                   << "num_void_args = " << num_void_args_ << ", "
                   << "func_name = " << func_name_ << ", "
                   << "lib_name = " << lib_name << "\n";
                os << "// CEXT Source \n"
                   << "// -------------\n"
                   << c_merged_code;
                
                LOG(FATAL) << os.str();
            }
        }

    private:
        // internal module
        CEXTModuleNode* m_;
        // the resource holder
        ObjectPtr<Object> sptr_;
        // the name of the function
        std::string func_name_; 
        // the number of args
        size_t num_void_args_;
        // the type of args
        std::vector<DLDataType> arg_types_;
};

PackedFunc CEXTModuleNode::GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) {
    ICHECK_EQ(sptr_to_self.get(), this);
    ICHECK_NE(name, symbol::tvm_module_main) << "Device function do not have main";

    auto it = fmap_.find(name);
    if (it == fmap_.end())  return PackedFunc();
    const FunctionInfo& info = it->second;
    CEXTWrappedFunc f;
    f.Init(this, sptr_to_self, name, info.arg_types.size(), info.arg_types);
    // std::cout << "~~~~~ enter cext module to get function: " << name << "\n";
    return PackFuncVoidArgs(f, info.arg_types);
}

Module CEXTModuleCreate(std::string code, std::string clean_code, std::string fmt,
                        std::unordered_map<std::string, FunctionInfo> fmap) {
    auto n = make_object<CEXTModuleNode>(code, clean_code, fmt, fmap);
    return Module(n);
}

}   // namespace runtime    
}   // namespace tvm