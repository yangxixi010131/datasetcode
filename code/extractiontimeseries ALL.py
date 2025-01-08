from nilearn import maskers, datasets
from nilearn.interfaces.fmriprep import load_confounds_strategy
# pickle是一个将Python对象保存为文件的模块
import pickle
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
                    if item.startswith('sub-') and os.path.isdir(os.path.join(dir_path, item)):  
                        yield from _find_files_recursive(os.path.join(dir_path, item), current_level + 1, target_suffixes)  
                elif current_level == 2:  # 第二层，检查前缀  
                    # 假设第二层目录名称以'prefix2_'开始  
                    if item.startswith('ses-') and os.path.isdir(os.path.join(dir_path, item)):  
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

tsv_to_nii_mapping = {} 
for file_path in files_found:  
    # print(file_path)
# 遍历文件路径列表    
    # 检查文件是否是时间序列文件（.tsv）  
    if file_path.endswith('.tsv'):  
        tsv_to_nii_mapping[file_path]='sub'
    if file_path.endswith('.gz'):
        tsv_to_nii_mapping[A] = file_path
    A=file_path
# print(tsv_to_nii_mapping)      
# 遍历加载数据
# data_path = r"E:\datasets\PPMI_fMRI\derivatives\sub-3378\ses-4\func\sub-3378_ses-4_task-rest_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
# confounds_path = r"E:\datasets\PPMI_fMRI\derivatives\sub-3378\ses-4\func\sub-3378_ses-4_task-rest_run-01_desc-confounds_timeseries.tsv"

# 加载ALL脑图谱模板
# load AAL atlas template.
atlas = datasets.fetch_atlas_aal()
# Loading atlas image stored in 'maps'
atlas_filename = atlas['maps']

# 创建ROI遮罩，以便从每个ROI中提取时间序列
masker = maskers.NiftiLabelsMasker(labels_img=atlas_filename, standardize=True,
                                       memory='nilearn_cache', memory_level=1, verbose=5)
time_series_new = []

# 遍历字典的项（键和值）  
for confounds_path_key, data_path_value in tsv_to_nii_mapping.items():  
    # 将键添加到 confounds_paths 列表中（尽管这里可能命名为 'keys' 更为合适）  
    confounds_path = confounds_path_key  
    # 将值添加到 data_paths 列表中  
    data_path = data_path_value 
    # print(f"confounds_path:{confounds_path},data_path{data_path}")  

# 提取时间序列，data是BOLD数据，confounds是噪音
    confounds, sample_mask = load_confounds_strategy(data_path)
    time_series = masker.fit_transform(data_path, confounds=confounds, sample_mask=sample_mask)
    # print(time_series)
# 提取出来的时间序列都存到一个新的列表中
    time_series_new.append(time_series) 
# 保存为pickle文件
with open("time_series_test.pkl_new", "wb") as f:
    pickle.dump(time_series_new, f)

# 读取pickle文件
with open("time_series_test.pkl_new", "rb") as f:
    time_series_new = pickle.load(f)