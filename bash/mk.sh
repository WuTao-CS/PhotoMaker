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