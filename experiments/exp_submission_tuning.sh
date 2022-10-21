#python classifier.py --tensorboard_log_dir experiments/exp_baseline_submission --seed 100 --split_val_fold_idx 0 --split_test_prop 0 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.2 --predict_data_dir data/subm_cfp_od_crop_OD_f2.0
#python classifier.py --tensorboard_log_dir experiments/exp_baseline_submission --seed 200 --split_val_fold_idx 1 --split_test_prop 0 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.2 --predict_data_dir data/subm_cfp_od_crop_OD_f2.0
#python classifier.py --tensorboard_log_dir experiments/exp_baseline_submission --seed 300 --split_val_fold_idx 2 --split_test_prop 0 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.2 --predict_data_dir data/subm_cfp_od_crop_OD_f2.0
#python classifier.py --tensorboard_log_dir experiments/exp_baseline_submission --seed 400 --split_val_fold_idx 3 --split_test_prop 0 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.2 --predict_data_dir data/subm_cfp_od_crop_OD_f2.0
#python classifier.py --tensorboard_log_dir experiments/exp_baseline_submission --seed 500 --split_val_fold_idx 4 --split_test_prop 0 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.2 --predict_data_dir data/subm_cfp_od_crop_OD_f2.0

#python classifier.py --tensorboard_log_dir experiments/exp_submission_tuning --seed 100 --split_val_fold_idx 0 --split_test_prop 0 --lr 7.03125e-5 --batch_size 90 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.6 --predict_data_dir data/subm_cfp_od_crop_OD_f2.0
#python classifier.py --tensorboard_log_dir experiments/exp_submission_tuning --seed 200 --split_val_fold_idx 1 --split_test_prop 0 --lr 7.03125e-5 --batch_size 90 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.6 --predict_data_dir data/subm_cfp_od_crop_OD_f2.0
#python classifier.py --tensorboard_log_dir experiments/exp_submission_tuning --seed 300 --split_val_fold_idx 2 --split_test_prop 0 --lr 7.03125e-5 --batch_size 90 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.6 --predict_data_dir data/subm_cfp_od_crop_OD_f2.0
#python classifier.py --tensorboard_log_dir experiments/exp_submission_tuning --seed 400 --split_val_fold_idx 3 --split_test_prop 0 --lr 7.03125e-5 --batch_size 90 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.6 --predict_data_dir data/subm_cfp_od_crop_OD_f2.0
#python classifier.py --tensorboard_log_dir experiments/exp_submission_tuning --seed 500 --split_val_fold_idx 4 --split_test_prop 0 --lr 7.03125e-5 --batch_size 90 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.6 --predict_data_dir data/subm_cfp_od_crop_OD_f2.0

#python classifier.py --tensorboard_log_dir experiments/exp_submission_tuning2 --seed 100 --split_val_fold_idx 0 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5 --aug_scale 0.1
#python classifier.py --tensorboard_log_dir experiments/exp_submission_tuning2 --seed 200 --split_val_fold_idx 1 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5 --aug_scale 0.1
#python classifier.py --tensorboard_log_dir experiments/exp_submission_tuning2 --seed 300 --split_val_fold_idx 2 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5 --aug_scale 0.1
#python classifier.py --tensorboard_log_dir experiments/exp_submission_tuning2 --seed 400 --split_val_fold_idx 3 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5 --aug_scale 0.1
#python classifier.py --tensorboard_log_dir experiments/exp_submission_tuning2 --seed 500 --split_val_fold_idx 4 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5 --aug_scale 0.1

python classifier.py --tensorboard_log_dir experiments/exp_submission_tuning2 --seed 100 --split_val_fold_idx 0 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.6 --aug_scale 0.1
python classifier.py --tensorboard_log_dir experiments/exp_submission_tuning2 --seed 200 --split_val_fold_idx 1 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.6 --aug_scale 0.1
python classifier.py --tensorboard_log_dir experiments/exp_submission_tuning2 --seed 300 --split_val_fold_idx 2 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.6 --aug_scale 0.1
python classifier.py --tensorboard_log_dir experiments/exp_submission_tuning2 --seed 400 --split_val_fold_idx 3 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.6 --aug_scale 0.1
python classifier.py --tensorboard_log_dir experiments/exp_submission_tuning2 --seed 500 --split_val_fold_idx 4 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.6 --aug_scale 0.1

python classifier.py --tensorboard_log_dir experiments/exp_submission_tuning2 --seed 100 --split_val_fold_idx 0 --split_test_prop 0 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5 --aug_scale 0.1 --predict_data_dir data/subm_cfp_od_crop_OD_f2.0
python classifier.py --tensorboard_log_dir experiments/exp_submission_tuning2 --seed 200 --split_val_fold_idx 1 --split_test_prop 0 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5 --aug_scale 0.1 --predict_data_dir data/subm_cfp_od_crop_OD_f2.0
python classifier.py --tensorboard_log_dir experiments/exp_submission_tuning2 --seed 300 --split_val_fold_idx 2 --split_test_prop 0 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5 --aug_scale 0.1 --predict_data_dir data/subm_cfp_od_crop_OD_f2.0
python classifier.py --tensorboard_log_dir experiments/exp_submission_tuning2 --seed 400 --split_val_fold_idx 3 --split_test_prop 0 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5 --aug_scale 0.1 --predict_data_dir data/subm_cfp_od_crop_OD_f2.0
python classifier.py --tensorboard_log_dir experiments/exp_submission_tuning2 --seed 500 --split_val_fold_idx 4 --split_test_prop 0 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5 --aug_scale 0.1 --predict_data_dir data/subm_cfp_od_crop_OD_f2.0

python classifier.py --tensorboard_log_dir experiments/exp_submission_tuning2 --seed 100 --split_val_fold_idx 0 --split_test_prop 0 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.6 --aug_scale 0.1 --predict_data_dir data/subm_cfp_od_crop_OD_f2.0
python classifier.py --tensorboard_log_dir experiments/exp_submission_tuning2 --seed 200 --split_val_fold_idx 1 --split_test_prop 0 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.6 --aug_scale 0.1 --predict_data_dir data/subm_cfp_od_crop_OD_f2.0
python classifier.py --tensorboard_log_dir experiments/exp_submission_tuning2 --seed 300 --split_val_fold_idx 2 --split_test_prop 0 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.6 --aug_scale 0.1 --predict_data_dir data/subm_cfp_od_crop_OD_f2.0
python classifier.py --tensorboard_log_dir experiments/exp_submission_tuning2 --seed 400 --split_val_fold_idx 3 --split_test_prop 0 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.6 --aug_scale 0.1 --predict_data_dir data/subm_cfp_od_crop_OD_f2.0
python classifier.py --tensorboard_log_dir experiments/exp_submission_tuning2 --seed 500 --split_val_fold_idx 4 --split_test_prop 0 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.6 --aug_scale 0.1 --predict_data_dir data/subm_cfp_od_crop_OD_f2.0