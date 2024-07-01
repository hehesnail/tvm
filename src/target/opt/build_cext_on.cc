/*!
 *  Build c extra modules from source.
 *
 * \file build_cext.cc
 */
#if defined(__linux__)
#include <sys/stat.h>
#endif
#include <cstdlib>

#include "../build_common.h"
#include "../source/codegen_c.h"
#include "../../runtime/cext/cext_module.h"

namespace tvm {
namespace codegen {

runtime::Module BuildCEXT(IRModule mod, Target target) {
    using tvm::runtime::Registry;
    std::string fmt = "c";

    // main source code generation
    bool output_ssa = false;
    CodeGenC cg;
    cg.Init(output_ssa);
    cg.EnableForParallel();

    Map<GlobalVar, PrimFunc> functions;
    for (auto [gvar, base_func] : mod->functions) {
        ICHECK(base_func->IsInstance<PrimFuncNode>()) << "CodeGenC: Can only take PrimFunc";
        auto prim_func = Downcast<PrimFunc>(base_func);
        auto calling_conv = prim_func->GetAttr<Integer>(tvm::attr::kCallingConv);
        ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
            << "CodeGenC: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
        functions.Set(gvar, prim_func);
    }

    for (auto [gvar, prim_func] : functions) {
        cg.DeclareFunction(gvar, prim_func);
    }
    for (auto [gvar, prim_func] : functions) {
        cg.AddFunction(gvar, prim_func);
    }

    // helper codegen to get clean source code
    CodeGenC cg_helper;
    cg_helper.Init(output_ssa);
    cg_helper.EnableForParallel();
    for (auto [gvar, prim_func] : functions) {
        cg_helper.NoDeclareFunction(gvar, prim_func);
    }
    for (auto [gvar, prim_func] : functions) {
        cg_helper.AddFunction(gvar, prim_func);
    }

    std::string code = cg.Finish();
    std::string clean_code = cg_helper.Finish();

    // std::cout << code << "\n";
    // std::cout << clean_code << "\n";

    // Hard-coded code for debug
    // code = R"(
    // })";

    // std::cout << "~~~~~~~~~~~~~~ -----> code \n";
    // std::cout << code << "\n";
    return CEXTModuleCreate(code, clean_code, fmt, ExtractFuncInfo(mod));    
}

TVM_REGISTER_GLOBAL("target.build.cext").set_body_typed(BuildCEXT);

}   // namespace codegen
}   // namespace tvm