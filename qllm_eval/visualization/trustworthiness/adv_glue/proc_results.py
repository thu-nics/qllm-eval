import os
import shutil

root_path = 'outputs'
out_path = 'assets'

os.mkdir(out_path)

for model in os.listdir(root_path):
    os.mkdir(os.path.join(out_path, model))
    for precision in os.listdir(os.path.join(root_path, model)):
        for i in range(len(os.listdir(os.path.join(root_path, model, precision)))):
            timestamp = os.listdir(os.path.join(root_path, model, precision))[i]
            if os.path.exists(os.path.join(root_path, model, precision, timestamp, 'summary')):
                break
        for file_name in os.listdir(os.path.join(root_path, model, precision, timestamp, 'summary')):
            if file_name.endswith('.csv'):
                break
        src_file = os.path.join(root_path, model, precision, timestamp, 'summary', file_name)
        dst_file = os.path.join(out_path, model, f'{precision}.csv')
        shutil.copy(src_file, dst_file)
        

        

        
