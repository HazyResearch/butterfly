#time OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 imagenet_main.py /data/imagenet --data-backend dali-gpu -a mobilenetv1 --width 0.5 --struct odo_4_res --sm-pooling 2 --n-struct-layers 7 --workers 12 | tee mobilenetv1_0.5_odo4res_smpool_2.log; mkdir -p ~/imagenet_training/mobilenetv1_0.5_odo4res_smpool_2 && mv -f mobilenetv1_0.5_odo4res_smpool_2.log experiment_raport.json checkpoint.pth.tar model_best.pth.tar ~/imagenet_training/mobilenetv1_0.5_odo4res_smpool_2/

# Sparse w/ channel pooling before softmax
result_name='mobilenetv1_0.5_sparse_smpool_2'
time OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 imagenet_main.py /data/imagenet --data-backend dali-gpu -a mobilenetv1 --width 0.5 --sparse --sm-pooling 2 --n-struct-layers 7 --workers 12 | tee ${result_name}.log; mkdir -p ~/imagenet_training/$result_name && mv -f ${result_name}.log experiment_raport.json checkpoint.pth.tar model_best.pth.tar ~/imagenet_training/$result_name


#time OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 imagenet_main.py /data/imagenet --data-backend dali-gpu --amp --dynamic-loss-scale -a mobilenetv1 --width 1.0 --struct odo_4_res --sm-pooling 2 --n-struct-layers 7 --workers 12 | tee mobilenetv1_odo4res_smpool_2.log; mkdir -p ~/imagenet_training/mobilenetv1_odo4res_smpool_2 && mv -f mobilenetv1_odo4res_smpool_2.log experiment_raport.json checkpoint.pth.tar model_best.pth.tar ~/imagenet_training/mobilenetv1_odo4res_smpool_2/

# width 1
result_name='mobilenetv1_sparse_smpool_2'
time OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 imagenet_main.py /data/imagenet --data-backend dali-gpu --amp --dynamic-loss-scale -a mobilenetv1 --width 1.0 --sparse --sm-pooling 2 --n-struct-layers 7 --workers 12 | tee ${result_name}.log; mkdir -p ~/imagenet_training/$result_name && mv -f ${result_name}.log experiment_raport.json checkpoint.pth.tar model_best.pth.tar ~/imagenet_training/$result_name


#time OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 imagenet_main.py /data/imagenet --data-backend dali-cpu --amp --dynamic-loss-scale -b 224 -a mobilenetv1 --width 2.0 --struct odo_4_res --sm-pooling 2 --n-struct-layers 7 --workers 12 | tee mobilenetv1_2.0_odo4res_smpool_2.log; mkdir -p ~/imagenet_training/mobilenetv1_2.0_odo4res_smpool_2 && mv -f mobilenetv1_2.0_odo4res_smpool_2.log experiment_raport.json checkpoint.pth.tar model_best.pth.tar ~/imagenet_training/mobilenetv1_2.0_odo4res_smpool_2/

# width 2
result_name='mobilenetv1_2.0_sparse_smpool_2'
time OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 imagenet_main.py /data/imagenet --data-backend dali-cpu --amp --dynamic-loss-scale -b 224 -a mobilenetv1 --width 2.0 --sparse --sm-pooling 2 --n-struct-layers 7 --workers 12 | tee ${result_name}.log; mkdir -p ~/imagenet_training/$result_name && mv -f ${result_name}.log experiment_raport.json checkpoint.pth.tar model_best.pth.tar ~/imagenet_training/$result_name
