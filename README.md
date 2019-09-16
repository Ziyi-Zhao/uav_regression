# uav_regression
The code repository for the UAV density regression task

# training parameters
--data_path\
/home/zzhao/UAV_POSTPROCESS/data/training_data_trajectory.npy

--structure\
basic_cnn

--lr\
0.001

--momentum\
0.9

--weight_decay\
0.1

--batch_size\
32

--num_epochs\
50

--split_ratio\
0.9

--checkpoint_dir\
/home/zzhao/data/uav_regression/check_point

--model_checkpoint_name\
uav_regression
