import os
import subprocess

# 定义源代码目录和目标目录
source_dir = 'save_cpp'
target_dir = 'save_ll'
include_dirs = ['-I', '/root/codegen_tvm/tvm/include/', '-I', '/root/codegen_tvm/tvm/3rdparty/dlpack/include/']

# 如果目标目录不存在，则创建它
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 初始化计数器和失败文件列表
success_count = 0
fail_count = 0
fail_files = []

# 遍历目录下的所有cpp文件
for filename in os.listdir(source_dir):
    if filename.endswith('.c'):
        # 构建源文件和目标文件的完整路径
        source_file = os.path.join(source_dir, filename)
        target_file = os.path.join(target_dir, os.path.splitext(filename)[0] + '.ll')
        
        # 构建clang命令
        command = ['clang', '-S', '-emit-llvm', source_file, '-o', target_file] + include_dirs
        
        # 执行命令
        try:
            subprocess.run(command, check=True)
            print(f"build {filename} success, file path: {target_file}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"build {filename} faild：{e}")
            fail_count += 1
            fail_files.append(filename)

print(f"successful: {success_count} files")
print(f"fail:       {fail_count} files")

if fail_count > 0:
    print("Files that failed to compile:")
    for file in fail_files:
        print(file)
