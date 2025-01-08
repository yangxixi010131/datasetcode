import os  
  
def find_files_with_suffixes_in_fourth_level(root_dir, target_suffixes):  
    # 递归函数，用于在指定层级查找具有特定后缀的文件  
    def _find_files_recursive(dir_path, current_level, target_suffixes):  
        # 检查当前层级  
        if current_level == 4:  # 第四层  
            # 查找具有特定后缀的文件  
            for item in os.listdir(dir_path):  
                item_path = os.path.join(dir_path, item)  
                if os.path.isfile(item_path) and any(item.endswith(suffix) for suffix in target_suffixes):  
                    yield item_path  # 使用生成器返回找到的文件路径  
        else:  
            # 遍历当前目录下的所有文件和文件夹  
            for item in os.listdir(dir_path):  
                # 根据层级应用不同的条件  
                if current_level == 1:  # 第一层，检查前缀  
                    # 假设第一层目录名称以'prefix1_'开始  
                    if item.startswith('sub-3351') and os.path.isdir(os.path.join(dir_path, item)):  
                        yield from _find_files_recursive(os.path.join(dir_path, item), current_level + 1, target_suffixes)  
                elif current_level == 2:  # 第二层，检查前缀  
                    # 假设第二层目录名称以'prefix2_'开始  
                    if item.startswith('ses-1') and os.path.isdir(os.path.join(dir_path, item)):  
                        yield from _find_files_recursive(os.path.join(dir_path, item), current_level + 1, target_suffixes)  
                elif current_level == 3:  # 第三层，检查名称（这里假设名称相同）  
                    # 假设第三层目录名称是固定的，例如'target_dir_name'  
                    if item == 'func' and os.path.isdir(os.path.join(dir_path, item)):  
                        yield from _find_files_recursive(os.path.join(dir_path, item), current_level + 1, target_suffixes)  
  
    # 开始搜索  
    results = []  
    for file_path in _find_files_recursive(root_dir, 1, target_suffixes):  
        results.append(file_path)  
    return results  
  
# 使用示例  
root_dir = 'E:\datasets\PPMI_fMRI\derivatives'  # 替换为你的根目录路径  
target_suffixes = ['_desc-preproc_bold.nii.gz', '_desc-confounds_timeseries.tsv']  # 替换为你要查找的文件后缀列表  
files_found = find_files_with_suffixes_in_fourth_level(root_dir, target_suffixes)  
for file_path in files_found:  
    print(file_path)

# import os  
  
# def find_files_with_suffix(root_dir, prefixes, target_suffix, level_to_check):  
#     # 初始化一个空列表来存储找到的文件  
#     files_found = []  
  
#     def _find_files_recursive(dir_path, current_level, current_prefix):  
#         # 如果当前层级是我们要检查的层级  
#         if current_level == level_to_check:  
#             for item in os.listdir(dir_path):  
#                 item_path = os.path.join(dir_path, item)  
#                 if os.path.isfile(item_path) and item.endswith(target_suffix):  
#                     files_found.append(item_path)  
#             return  
  
#         # 遍历当前目录下的所有文件和文件夹  
#         for item in os.listdir(dir_path):  
#             item_path = os.path.join(dir_path, item)  
#             if os.path.isdir(item_path):  
#                 # 检查层级和前缀  
#                 if current_level == 1 and item.startswith(prefixes[0]):  
#                     _find_files_recursive(item_path, current_level + 1, prefixes[0])  
#                 elif current_level == 2 and item.startswith(prefixes[1]):  
#                     _find_files_recursive(item_path, current_level + 1, prefixes[1])  
#                 elif current_level == 3 and item == prefixes[2]:  # 假设第三层名称相同  
#                     _find_files_recursive(item_path, current_level + 1, '')  # 第三层之后不再需要前缀检查  
  
#     # 开始搜索  
#     _find_files_recursive(root_dir, 1, prefixes[0])  
  
#     # 返回找到的文件列表  
#     return files_found  
  
# # 使用示例  
# root_dir = 'E:\datasets\PPMI_fMRI\derivatives'  # 替换为你的根目录路径  
# prefixes = ('sub-', 'ses-', 'func')  # 前三层的前缀  
# suffix1 = '_desc-preproc_bold.nii.gz'  # 第一种后缀  
# suffix2 = '_desc-confounds_timeseries.tsv'  # 第二种后缀  
  
# # 查找并打印第一种后缀的文件  
# files_with_suffix1 = find_files_with_suffix(root_dir, prefixes, suffix1, 4)  
# print("Files with suffix _desc-preproc_bold.nii.gz:")  
# for file_path in files_with_suffix1:  
#     print(file_path)  
  
# # 查找并打印第二种后缀的文件  
# files_with_suffix2 = find_files_with_suffix(root_dir, prefixes, suffix2, 4)  
# print("\nFiles with suffix _desc-confounds_timeseries.tsv:")  
# for file_path in files_with_suffix2:  
#     print(file_path)
