#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <string>
#include <vector>
#include <any>

#include <cstdarg>
#include <dlfcn.h>

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
    std::string command = "gcc -fopenmp -pthread -fPIC -shared -o " + so_file_path.string() + " " + c_file_path.string();

    int ret = system(command.c_str());
    if (ret != 0) {
        throw std::runtime_error("Compilation failed. " + command);
    }

    return so_file_path.string();
}

std::string CompilePathFile(const std::string& path, const std::string& file, std::string& name) {
    std::filesystem::path dir_path = path;
    std::filesystem::path path_to_file = dir_path / file;

    // check file exist
    if (!std::filesystem::exists(path_to_file)) {
        throw std::runtime_error("Path to file does not exist: " + path_to_file.string());
    }

    // create shared lib path
    std::filesystem::path so_file_path = dir_path / ("lib" + name + ".so");
    std::string command = "gcc -fopenmp -pthread -fPIC -shared -o " + so_file_path.string() + " " + path_to_file.string();

    int ret = system(command.c_str());
    if (ret != 0) {
        throw std::runtime_error("Compilation failed. " + command);
    }

    return so_file_path.string();
}

std::string GenBridgeCode(int num_args, const std::string& name) {
    std::ostringstream oss;
    oss << "\nvoid bridge_call(void** void_args) {\n";
    oss << "    " + name + "(";
    for (int i = 0; i < num_args - 1; i++) {
        oss << "void_args[" + std::to_string(i) + "], ";
    }
    oss << "void_args[" + std::to_string(num_args-1) + "]";
    oss << ");\n";
    oss << "}\n";
    return oss.str();
}

std::string CompileSimpleFunction(int num_args, const std::string& kernel_name, const std::string& path) {
    std::string code = R"(
        void add(float *a, float *b) {
            b[0] = a[0] + 1;
        }
    )";
    try { 
        auto bridge_code = GenBridgeCode(num_args, kernel_name);
        auto combine_code = code + bridge_code;
        std::cout << combine_code << "\n";
        auto ret = CompileStringCode(combine_code, kernel_name, path);
        return ret;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
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
        return CEXTResult::CEXT_LOAD_FUNC_SYMBOL_FAIL;
    }

    // utilize bridge_call func to call kernel func
    bridge_call_func(void_args);

    // close dynamic library
    dlclose(handle);

    return CEXTResult::CEXT_SUCCESS;
}

int main() {

    std::string kernel_name = "add";
    std::string path = "./tmp";
    std::string lib_name = CompileSimpleFunction(2, kernel_name, path);

    void** void_args = static_cast<void**>(malloc(2 * sizeof(void*)));
    float *a = new float(10);
    float *b = new float(20);

    void_args[0] = a;
    void_args[1] = b;
    cextLaunchKernel(void_args, kernel_name, lib_name);

    std::cout << "Result: " << *b << std::endl;

    return 0;
} 