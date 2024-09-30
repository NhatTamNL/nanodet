import os
import shutil

def move_files(txt_file, dest_dir):
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            file_path = line.strip()
            if os.path.exists(file_path):
                dest_path = os.path.join(dest_dir, os.path.basename(file_path))
                shutil.move(file_path, dest_path)
                print(f"Moved {file_path} to {dest_path}")
            else:
                print(f"File {file_path} does not exist")

# Tạo các thư mục đích nếu chưa tồn tại
os.makedirs('train/images', exist_ok=True)
# os.makedirs('val/images', exist_ok=True)
# os.makedirs('test/images', exist_ok=True)

# Di chuyển các tệp
move_files('train.txt', 'train/images')
# move_files('valid.txt', 'val/images')
# move_files('test.txt', 'test/labels')
