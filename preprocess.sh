#!/bin/bash

# 路径配置
ABC_EXE="./lib/abc/abc"
DATASET="ISCAS89"
INPUT_DIR="./benchmarks/$DATASET"
OUTPUT_DIR="./benchmarks/${DATASET}_cleaned"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# # 遍历输入目录下的所有 .v 文件
# for file in $INPUT_DIR/*.v; do
#     filename=$(basename "$file")
#     output_file="$OUTPUT_DIR/$filename"
    
#     echo "Processing $filename ..."
    
#     # ABC 命令解释：
#     # read_verilog: 读取原始 Verilog
#     # strash: 转换为 AIG (结构化哈希)
#     # write_verilog: 输出为标准的 assign 格式
#     # $ABC_EXE -c "read_verilog $file; strash; write_verilog $output_file"
#     $ABC_EXE" -c "read_verilog -nocase -ignore_warnings $file; strash; clean; write_verilog -noexpr -simple $output_file
# done

for file in $INPUT_DIR/*.v; do
    filename=$(basename "$file")
    output_file="$OUTPUT_DIR/$filename"
    echo "Cleaning $filename with Yosys..."
    
    # 使用 Yosys 清洗 -> 变成最纯粹的门级网表
    yosys -p "read_verilog $file; synth; abc -g AND,OR,XOR; write_verilog $output_file" > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "Success: $output_file"
    else
        echo "Failed: $filename"
    fi
done

echo "Done! Cleaned benchmarks are in $OUTPUT_DIR"
