#!/bin/bash

# 定义起始和结束的数字
start=6
end=16

# 循环创建文件
for ((i=start; i<=end; i++))
do
    idx=$((i-1))
    filename="${i}_test.sh"
    touch "$filename"
    echo "#!/bin/bash" > "$filename"
    echo "python process_reg_data.py --phase ${idx}" >> "$filename"
    chmod +x "$filename"
done

echo "Files created successfully."


CUDA_VISIBLE_DEVICES=0 python preprocess_sd15_new_face.py --phase 0 --total 4
CUDA_VISIBLE_DEVICES=1 python preprocess_sd15_new_face.py --phase 1 --total 4
CUDA_VISIBLE_DEVICES=2 python preprocess_sd15_new_face.py --phase 2 --total 4
CUDA_VISIBLE_DEVICES=3 python preprocess_sd15_new_face.py --phase 3 --total 4

CUDA_VISIBLE_DEVICES=0 python preprocess_sd15_new_face.py --phase 4
CUDA_VISIBLE_DEVICES=1 python preprocess_sd15_new_face.py --phase 5
CUDA_VISIBLE_DEVICES=0 python preprocess_sd15_new_face.py --phase 6
CUDA_VISIBLE_DEVICES=3 python preprocess_sd15_new_face.py --phase 7