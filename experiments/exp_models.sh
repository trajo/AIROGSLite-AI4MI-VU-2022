python classifier.py --tensorboard_log_dir exp_models --seed 100 --split_val_fold_idx 0 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone google/vit-base-patch32-384 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5
python classifier.py --tensorboard_log_dir exp_models --seed 200 --split_val_fold_idx 1 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone google/vit-base-patch32-384 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5
python classifier.py --tensorboard_log_dir exp_models --seed 300 --split_val_fold_idx 2 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone google/vit-base-patch32-384 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5

python classifier.py --tensorboard_log_dir exp_models --seed 100 --split_val_fold_idx 0 --split_test_prop 0.2 --lr 1.857e-5 --batch_size 24 --backbone microsoft/swin-base-patch4-window12-384-in22k --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5
python classifier.py --tensorboard_log_dir exp_models --seed 200 --split_val_fold_idx 1 --split_test_prop 0.2 --lr 1.857e-5 --batch_size 24 --backbone microsoft/swin-base-patch4-window12-384-in22k --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5
python classifier.py --tensorboard_log_dir exp_models --seed 300 --split_val_fold_idx 2 --split_test_prop 0.2 --lr 1.857e-5 --batch_size 24 --backbone microsoft/swin-base-patch4-window12-384-in22k --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5

python classifier.py --tensorboard_log_dir exp_models --seed 100 --split_val_fold_idx 0 --split_test_prop 0.2 --lr 5e-5 --batch_size 1.25e-5 --backbone microsoft/swin-large-patch4-window12-384-in22k --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5
python classifier.py --tensorboard_log_dir exp_models --seed 200 --split_val_fold_idx 1 --split_test_prop 0.2 --lr 5e-5 --batch_size 1.25e-5 --backbone microsoft/swin-large-patch4-window12-384-in22k --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5
python classifier.py --tensorboard_log_dir exp_models --seed 300 --split_val_fold_idx 2 --split_test_prop 0.2 --lr 5e-5 --batch_size 1.25e-5 --backbone microsoft/swin-large-patch4-window12-384-in22k --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5

python classifier.py --tensorboard_log_dir exp_models --seed 100 --split_val_fold_idx 0 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224-vit_b_32.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5
python classifier.py --tensorboard_log_dir exp_models --seed 200 --split_val_fold_idx 1 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224-vit_b_32.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5
python classifier.py --tensorboard_log_dir exp_models --seed 300 --split_val_fold_idx 2 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224-vit_b_32.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5

python classifier.py --tensorboard_log_dir exp_models --seed 100 --split_val_fold_idx 0 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224vit_b_16.IMAGENET1K_SWAG_LINEAR_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5
python classifier.py --tensorboard_log_dir exp_models --seed 200 --split_val_fold_idx 1 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224vit_b_16.IMAGENET1K_SWAG_LINEAR_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5
python classifier.py --tensorboard_log_dir exp_models --seed 300 --split_val_fold_idx 2 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224vit_b_16.IMAGENET1K_SWAG_LINEAR_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5

python classifier.py --tensorboard_log_dir exp_models --seed 100 --split_val_fold_idx 0 --split_test_prop 0.2 --lr 2.5e-5 --batch_size 32 --backbone tv-384vit_b_16.IMAGENET1K_SWAG_E2E_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5
python classifier.py --tensorboard_log_dir exp_models --seed 200 --split_val_fold_idx 1 --split_test_prop 0.2 --lr 2.5e-5 --batch_size 32 --backbone tv-384vit_b_16.IMAGENET1K_SWAG_E2E_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5
python classifier.py --tensorboard_log_dir exp_models --seed 300 --split_val_fold_idx 2 --split_test_prop 0.2 --lr 2.5e-5 --batch_size 32 --backbone tv-384vit_b_16.IMAGENET1K_SWAG_E2E_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5

python classifier.py --tensorboard_log_dir exp_models --seed 100 --split_val_fold_idx 0 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224-resnext50_32x4d.IMAGENET1K_V2 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5
python classifier.py --tensorboard_log_dir exp_models --seed 200 --split_val_fold_idx 1 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224-resnext50_32x4d.IMAGENET1K_V2 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5
python classifier.py --tensorboard_log_dir exp_models --seed 300 --split_val_fold_idx 2 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224-resnext50_32x4d.IMAGENET1K_V2 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5

python classifier.py --tensorboard_log_dir exp_models --seed 100 --split_val_fold_idx 0 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5
python classifier.py --tensorboard_log_dir exp_models --seed 200 --split_val_fold_idx 1 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5
python classifier.py --tensorboard_log_dir exp_models --seed 300 --split_val_fold_idx 2 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.5
