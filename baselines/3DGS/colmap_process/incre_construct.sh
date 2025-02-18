#!/bin/bash

# 检查是否传入了两个参数
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_path> <destination_path>"
    exit 1
fi

# 定义传入的路径
SOURCE_PATH=$1
DEST_PATH=$2


# 拷贝数据库文件到指定的目标路径
cp "${SOURCE_PATH}/distorted/database.db" "${DEST_PATH}"

# 执行 COLMAP 特征提取
colmap feature_extractor \
    --database_path "${DEST_PATH}/database.db" \
    --image_path "${DEST_PATH}/input" \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.mask_path "${DEST_PATH}/masks" \
    --image_list_path "${DEST_PATH}/test_set.txt" \
    --SiftExtraction.use_gpu 1 \
    --ImageReader.single_camera 1

# 执行 COLMAP 词汇树匹配
colmap vocab_tree_matcher \
    --database_path "${DEST_PATH}/database.db" \
    --VocabTreeMatching.vocab_tree_path colmap_process/vocab_tree_flickr100K_words256K.bin \
    --VocabTreeMatching.match_list_path "${DEST_PATH}/test_set.txt"

# 创建稀疏模型输出目录
mkdir -p "${DEST_PATH}/sparse/0"

# 执行 COLMAP 3D 重建 (Mapper)
colmap mapper \
    --database_path "${DEST_PATH}/database.db" \
    --image_path "${DEST_PATH}/input" \
    --input_path "${SOURCE_PATH}/sparse/0" \
    --output_path "${DEST_PATH}/sparse/0"