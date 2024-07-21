import os
from saliency_toolbox import calculate_measures


# 1: 修改work_name
work_name = 'work_4'

# 2: 修改model_tag
model_tag = 'Net_main'

# 3: 修改测试集
train_tag_list = ['D']
val_tag_list = ['DAVIS', 'DUT_OMRON', 'DUTS', 'HRSOD', 'UHRSD']


root_path = os.path.join('/data2021/tb/AllWork', work_name, 'Detect/Results')
results_path = os.path.join(root_path, model_tag)
if not os.path.exists(results_path):
    os.mkdir(results_path)


for train_tag in train_tag_list:
    for val_tag in val_tag_list:
        
        file_name = train_tag + '_' + val_tag + '.txt'
        results_file = os.path.join(results_path, file_name)

        metric_list = ['MAE', 'Max-F', 'E-measure', 'S-measure']
        values = calculate_measures(work_name=work_name, model_tag=model_tag, val_tag=val_tag, measures=metric_list)

        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            write_info = f"Fm:{values['Max-F']:.3f}    MAE:{values['MAE']:.3f}    Em:{values['E-measure']:.3f}    Sm:{values['S-measure']:.3f}\n"
            f.write(write_info)

