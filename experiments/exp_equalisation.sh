python classifier.py --seed 1 --split_val_fold_idx 0 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.2 --aug_hist_equalize yes
python classifier.py --seed 1 --split_val_fold_idx 0 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.2 --aug_hist_equalize IgnoreBlack

#python classifier.py --seed 2 --split_val_fold_idx 1 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.2 --aug_hist_equalize yes
#python classifier.py --seed 2 --split_val_fold_idx 1 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.2 --aug_hist_equalize IgnoreBlack

#python classifier.py --seed 3 --split_val_fold_idx 2 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.2 --aug_hist_equalize yes
#python classifier.py --seed 3 --split_val_fold_idx 2 --split_test_prop 0.2 --lr 5e-5 --batch_size 64 --backbone tv-224-swin_b.IMAGENET1K_V1 --data_dir ./data/cfp_od_crop_OD_f2.0 --optimizer adamw --od_crop_factor 1.2 --aug_hist_equalize IgnoreBlack
