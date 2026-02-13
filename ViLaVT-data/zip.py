import os
import zipfile
import re

def parse_size(size_input):
    """
    将带单位的大小字符串转换为字节数。
    支持单位：KB、MB、GB（不区分大小写）。
    如果不带单位，则视为字节数。
    """
    if isinstance(size_input, int):
        return size_input
    size_input = size_input.strip().upper()
    match = re.match(r"(\d+(\.\d+)?)(KB|MB|GB)?", size_input)
    if not match:
        raise ValueError("无效的大小格式")
    number = float(match.group(1))
    unit = match.group(3)
    if unit == "GB":
        return int(number * 1024**3)
    elif unit == "MB":
        return int(number * 1024**2)
    elif unit == "KB":
        return int(number * 1024)
    else:
        return int(number)

def split_and_zip(folder, output_prefix, max_size):
    """
    将 folder 内的所有文件按文件大小分组压缩，
    每个压缩包最大大小为 max_size 字节，
    生成的压缩包命名为 output_prefix_part1.zip, output_prefix_part2.zip, ...，
    并且所有压缩包均为独立的 ZIP 文件（扩展名都是 .zip）。
    """
    # 遍历文件夹，获得所有文件的绝对路径及大小
    files = []
    for root, dirs, filenames in os.walk(folder):
        print(root, dirs, filenames)
        for filename in filenames:
            file_path = os.path.join(root, filename)
            size = os.path.getsize(file_path)
            files.append((file_path, size))
    
    # 按顺序将文件分组，确保每组总大小不超过 max_size
    groups = []
    current_group = []
    current_group_size = 0
    for file_path, size in files:
        # 如果当前组不为空且加上当前文件会超出 max_size，则另起一组
        if current_group and current_group_size + size > max_size:
            groups.append(current_group)
            current_group = []
            current_group_size = 0
        current_group.append(file_path)
        current_group_size += size
    if current_group:
        groups.append(current_group)
    
    # 分别生成每个组的 ZIP 文件
    for i, group in enumerate(groups, start=1):
        zip_filename = f"{output_prefix}_part{i}.zip"
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in group:
                # 保留相对路径
                arcname = os.path.relpath(file_path, folder)
                zipf.write(file_path, arcname)
        print(f"Created {zip_filename} with {len(group)} files.")

if __name__ == "__main__":
    # 请根据需要修改以下参数：
    # folder = "./rl"         # 待压缩的文件夹路径（使用原始字符串避免转义问题）
    # output_prefix = "rl"                # 压缩包输出的前缀
    folder = "./sft/SPAR7M_TEXT"
    output_prefix = "SPAR7M_TEXT"
    max_size = parse_size("5GB")           # 每个压缩包最大 1GB，可以修改为 "500MB"、"2GB" 等格式

    split_and_zip(folder, output_prefix, max_size)
