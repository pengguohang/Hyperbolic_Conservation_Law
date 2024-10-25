# 测试几种loss
python train_com_new.py --epochs 10000 --model_name PINNs --activate Tanh --range 0.1 --param 256 --layer 8 --model MLP --sample 1
python train_com_new.py --epochs 30000 --model_name PINNs --activate Tanh --range 0.1 --param 256 --layer 8 --model MLP --sample 1

python train_com_new.py --epochs 10000 --model_name DFVM --activate Tanh --range 0.1 --param 256 --layer 8 --model MLP --sample 1
python train_com_new.py --epochs 30000 --model_name DFVM --activate Tanh --range 0.1 --param 256 --layer 8 --model MLP --sample 1

python train_com_new.py --epochs 10000 --model_name MIX_all --activate Tanh --range 0.1 --param 256 --layer 8 --model MLP --sample 1
python train_com_new.py --epochs 30000 --model_name MIX_all --activate Tanh --range 0.1 --param 256 --layer 8 --model MLP --sample 1

python train_com_new.py --epochs 10000 --model_name MIX_region --activate Tanh --range 0.1 --param 256 --layer 8 --model MLP --sample 1
python train_com_new.py --epochs 30000 --model_name MIX_region --activate Tanh --range 0.1 --param 256 --layer 8 --model MLP --sample 1



# python train_com_new.py --epochs 10000 --model_name MIX_all --activate Tanh --range 0.1 --param 128 --layer 4 --model MLP --sample QMC
# python train_com_new.py --epochs 30000 --model_name MIX_all --activate Tanh --range 0.1 --param 128 --layer 4 --model MLP --sample QMC
# python train_com_new.py --epochs 10000 --model_name MIX_region --activate Tanh --range 0.1 --param 128 --layer 4 --model MLP --sample QMC
# python train_com_new.py --epochs 30000 --model_name MIX_region --activate Tanh --range 0.1 --param 128 --layer 4 --model MLP --sample QMC

# python train_com_new.py --epochs 10000 --model_name MIX_all --activate Tanh --range 0.1 --param 128 --layer 4 --model ResNet --sample QMC
# python train_com_new.py --epochs 30000 --model_name MIX_all --activate Tanh --range 0.1 --param 128 --layer 4 --model ResNet --sample QMC
# python train_com_new.py --epochs 10000 --model_name MIX_region --activate Tanh --range 0.1 --param 128 --layer 4 --model ResNet --sample QMC
# python train_com_new.py --epochs 30000 --model_name MIX_region --activate Tanh --range 0.1 --param 128 --layer 4 --model ResNet --sample QMC



# python train_com_new.py --epochs 500 --model_name PINNs --activate Tanh --range 0.1 --param 128 --layer 4 --model MLP --sample QMC
# python train_com_new.py --epochs 500 --model_name DFVM --activate Tanh --range 0.1 --param 128 --layer 4 --model MLP --sample QMC
# python train_com_new.py --epochs 500 --model_name MIX_region --activate Tanh --range 0.1 --param 128 --layer 4 --model ResNet --sample QMC
# python train_com_new.py --epochs 500 --model_name MIX --activate Tanh --range 0.1 --param 128 --layer 4 --model PINNsformer

python train_com_new.py --epochs 300 --model_name PINNs --activate Tanh --range 0.1 --param 128 --layer 4 --model PINNsformer --sample QMC