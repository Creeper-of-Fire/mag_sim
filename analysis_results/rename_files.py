import os

# --- 配置区 ---

# 1. 把这个脚本文件放到需要重命名文件的文件夹里。
# 2. 如果文件在其他文件夹，请修改下面的路径，例如 'C:/Users/YourUser/Desktop/files'
#    '.' 代表当前文件夹。
TARGET_DIRECTORY = '.'

# 3. 需要处理的文件名前缀列表 (注意要包含末尾的下划线 '_')
PREFIXES_TO_SWAP = [
    'comparison_baseline_',
    'analysis_energy_evolution_',
    'analysis_tail_stats_',
    'analysis_field_evolution_',
    'analysis_spectrum_',
    'video_field_slice_'
]


# --- 脚本核心逻辑 ---

def rename_files_in_directory(directory):
    """
    遍历指定目录，根据规则重命名文件。
    """
    print(f"开始扫描文件夹: '{os.path.abspath(directory)}'")

    # 获取目录下所有文件和文件夹的名字
    try:
        filenames = os.listdir(directory)
    except FileNotFoundError:
        print(f"错误：找不到文件夹 '{directory}'。请检查路径是否正确。")
        return

    renamed_count = 0
    for filename in filenames:
        # 构建完整的文件路径
        old_path = os.path.join(directory, filename)

        # 只处理文件，跳过文件夹
        if not os.path.isfile(old_path):
            continue

        # 遍历需要处理的前缀列表
        for prefix in PREFIXES_TO_SWAP:
            if filename.startswith(prefix):

                # 分离文件名和扩展名
                base_name, extension = os.path.splitext(filename)

                # 提取描述部分 (前缀之后的部分)
                description = base_name[len(prefix):]

                # 替换描述中的非法字符 '*' 为 'x'
                clean_description = description.replace('*', 'x')

                # 去掉前缀末尾的下划线，用于拼接到新名字的末尾
                prefix_for_suffix = prefix.strip('_')

                # 构建新的文件名
                new_filename = f"{clean_description}_{prefix_for_suffix}{extension}"

                # 构建新的完整路径
                new_path = os.path.join(directory, new_filename)

                # 打印将要执行的操作
                print(f"  - 准备重命名: '{filename}' -> '{new_filename}'")

                # 执行重命名
                try:
                    os.rename(old_path, new_path)
                    renamed_count += 1
                except OSError as e:
                    print(f"    重命名失败! 错误: {e}")

                # 匹配到一个前缀后，就不再继续匹配其他前缀了
                break

    print(f"\n处理完成！共重命名了 {renamed_count} 个文件。")


# --- 运行脚本 ---
if __name__ == "__main__":
    rename_files_in_directory(TARGET_DIRECTORY)