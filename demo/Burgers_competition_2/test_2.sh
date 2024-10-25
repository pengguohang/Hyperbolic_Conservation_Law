# # 测试几种loss
# python train_com_new.py --epochs 30000 --model_name GA --activate Tanh --range 0.1 --param 256 --layer 8 --model MLP --sample 1
# python train_com_new.py --epochs 30000 --model_name PINNs --activate Tanh --range 0.1 --param 256 --layer 8 --model MLP --sample 1
# python train_com_new.py --epochs 30000 --model_name DFVM --activate Tanh --range 0.1 --param 256 --layer 8 --model MLP --sample 1
# python train_com_new.py --epochs 30000 --model_name MIX_all --activate Tanh --range 0.1 --param 256 --layer 8 --model MLP --sample 1
# python train_com_new.py --epochs 30000 --model_name MIX_region --activate Tanh --range 0.1 --param 256 --layer 8 --model MLP --sample 1
# # 测试两种模型
# python train_com_new.py --epochs 30000 --model_name GA --activate Tanh --range 0.1 --param 256 --layer 8 --model ResNet --sample 1
# python train_com_new.py --epochs 30000 --model_name MIX_all --activate Tanh --range 0.1 --param 256 --layer 8 --model ResNet --sample 1
# python train_com_new.py --epochs 30000 --model_name MIX_region --activate Tanh --range 0.1 --param 256 --layer 8 --model ResNet --sample 1

# # 测试采样方式
# python train_com_new.py --epochs 30000 --model_name GA --activate Tanh --range 0.1 --param 256 --layer 8 --model ResNet --sample QMC
# python train_com_new.py --epochs 30000 --model_name MIX_all --activate Tanh --range 0.1 --param 256 --layer 8 --model ResNet --sample QMC
# python train_com_new.py --epochs 30000 --model_name MIX_region --activate Tanh --range 0.1 --param 256 --layer 8 --model ResNet --sample QMC

# # 不同架构大小
# python train_com_new.py --epochs 30000 --model_name GA --activate Tanh --range 0.1 --param 512 --layer 8 --model ResNet --sample QMC
# python train_com_new.py --epochs 30000 --model_name MIX_region --activate Tanh --range 0.1 --param 512 --layer 8 --model ResNet --sample QMC
# python train_com_new.py --epochs 30000 --model_name GA --activate Tanh --range 0.1 --param 256 --layer 10 --model ResNet --sample QMC
# python train_com_new.py --epochs 30000 --model_name MIX_region --activate Tanh --range 0.1 --param 256 --layer 10 --model ResNet --sample QMC

# # 不同区间大小
# python train_com_new.py --epochs 30000 --model_name MIX_region --activate Tanh --range 0.15 --param 256 --layer 10 --model ResNet --sample QMC
# python train_com_new.py --epochs 30000 --model_name MIX_region --activate Tanh --range 0.2 --param 256 --layer 10 --model ResNet --sample QMC
# python train_com_new.py --epochs 30000 --model_name MIX_region --activate Tanh --range 0.05 --param 256 --layer 10 --model ResNet --sample QMC

# # GA调参
# # 调大了参数，结果很差

# 加上entrop
python train_com_new.py --epochs 30000 --model_name GA --activate Tanh --range 0.1 --param 512 --layer 8 --model ResNet --sample QMC --entropy
python train_com_new.py --epochs 30000 --model_name MIX_all --activate Tanh --range 0.1 --param 512 --layer 8 --model ResNet --sample QMC --entropy
python train_com_new.py --epochs 30000 --model_name MIX_region --activate Tanh --range 0.1 --param 512 --layer 8 --model ResNet --sample QMC --entropy