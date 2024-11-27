#!/bin/bash

# 定义起始和结束的数字
start=31
end=38

# 循环创建文件
for ((i=start; i<=end; i++))
do
    idx=$((i-1))
    filename="${i}_test.sh"
    touch "$filename"
    chmod a+x "$filename"
done

echo "Files created successfully."


