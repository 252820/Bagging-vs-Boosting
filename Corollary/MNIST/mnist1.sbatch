#!/bin/bash

#SBATCH -J MNIST1 
#SBTACH -p A800
#SBATCH -N 1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=12
#SBATCH --output=result-%j.out           # 标准输出重定向到该文件
#SBATCH --error=error-%j.err             # 标准错误重定向到该文件
#SBATCH --qos qmultiple



# 加载任何必要的模块或环境
# module load python3

# 运行 Python 脚本
singularity exec my_python_env.sif python mnist1.py
