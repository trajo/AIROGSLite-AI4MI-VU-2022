# fine tuning the crop size
python classifier.py --tensorboard_log_dir exp_models --seed 100 --split_val_fold_idx 0 --split_test_prop 0.2 --lr 7.03125e-5 --batch_size 90 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5 --aug_rot_degrees 10 --aug_translate 0.2
python classifier.py --tensorboard_log_dir exp_models --seed 200 --split_val_fold_idx 1 --split_test_prop 0.2 --lr 7.03125e-5 --batch_size 90 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5 --aug_rot_degrees 10 --aug_translate 0.2
python classifier.py --tensorboard_log_dir exp_models --seed 300 --split_val_fold_idx 2 --split_test_prop 0.2 --lr 7.03125e-5 --batch_size 90 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5 --aug_rot_degrees 10 --aug_translate 0.2
python classifier.py --tensorboard_log_dir exp_models --seed 400 --split_val_fold_idx 3 --split_test_prop 0.2 --lr 7.03125e-5 --batch_size 90 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5 --aug_rot_degrees 10 --aug_translate 0.2
python classifier.py --tensorboard_log_dir exp_models --seed 500 --split_val_fold_idx 4 --split_test_prop 0.2 --lr 7.03125e-5 --batch_size 90 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5 --aug_rot_degrees 10 --aug_translate 0.2
#
python classifier.py --tensorboard_log_dir exp_models --seed 100 --split_val_fold_idx 0 --split_test_prop 0.2 --lr 7.03125e-5 --batch_size 90 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.6 --aug_rot_degrees 10 --aug_translate 0.2
python classifier.py --tensorboard_log_dir exp_models --seed 200 --split_val_fold_idx 1 --split_test_prop 0.2 --lr 7.03125e-5 --batch_size 90 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.6 --aug_rot_degrees 10 --aug_translate 0.2
python classifier.py --tensorboard_log_dir exp_models --seed 300 --split_val_fold_idx 2 --split_test_prop 0.2 --lr 7.03125e-5 --batch_size 90 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.6 --aug_rot_degrees 10 --aug_translate 0.2
python classifier.py --tensorboard_log_dir exp_models --seed 400 --split_val_fold_idx 3 --split_test_prop 0.2 --lr 7.03125e-5 --batch_size 90 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.6 --aug_rot_degrees 10 --aug_translate 0.2
python classifier.py --tensorboard_log_dir exp_models --seed 500 --split_val_fold_idx 4 --split_test_prop 0.2 --lr 7.03125e-5 --batch_size 90 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.6 --aug_rot_degrees 10 --aug_translate 0.2
#
python classifier.py --tensorboard_log_dir exp_models --seed 100 --split_val_fold_idx 0 --split_test_prop 0.2 --lr 7.03125e-5 --batch_size 90 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.4 --aug_rot_degrees 10 --aug_translate 0.2
python classifier.py --tensorboard_log_dir exp_models --seed 200 --split_val_fold_idx 1 --split_test_prop 0.2 --lr 7.03125e-5 --batch_size 90 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.4 --aug_rot_degrees 10 --aug_translate 0.2
python classifier.py --tensorboard_log_dir exp_models --seed 300 --split_val_fold_idx 2 --split_test_prop 0.2 --lr 7.03125e-5 --batch_size 90 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.4 --aug_rot_degrees 10 --aug_translate 0.2
python classifier.py --tensorboard_log_dir exp_models --seed 400 --split_val_fold_idx 3 --split_test_prop 0.2 --lr 7.03125e-5 --batch_size 90 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.4 --aug_rot_degrees 10 --aug_translate 0.2
python classifier.py --tensorboard_log_dir exp_models --seed 500 --split_val_fold_idx 4 --split_test_prop 0.2 --lr 7.03125e-5 --batch_size 90 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.4 --aug_rot_degrees 10 --aug_translate 0.2
