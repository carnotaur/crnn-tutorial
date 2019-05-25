import glob
import os 
import shutil
from sklearn.model_selection import train_test_split

all_pathes = glob.glob('data/samples/*')
train_pathes, test_pathes = train_test_split(all_pathes, random_state=123, test_size=0.2)

for orig_path in all_pathes:
    basename = os.path.basename(orig_path)
    
    if orig_path in train_pathes:
        dest_path = os.path.join('data/train', basename)
        shutil.move(orig_path, dest_path)
        
    elif orig_path in test_pathes:
        dest_path = os.path.join('data/val', basename)
        shutil.move(orig_path, dest_path)