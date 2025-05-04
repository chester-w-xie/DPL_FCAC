python Pre_train_FSC89.py -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 --trials 2 --im_pretrain \
-lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 10 -schedule Milestone -milestones 60 70 \
-gpu 1 -temperature 16 --metapath /SATA01/datasets/FSD-MIX-CLIPS-for_FSCIL/FSD_MIX_CLIPS.annotations_revised/FSC-89-meta \
--datapath /SATA01/datasets/FSD-MIX-CLIPS-for_FSCIL/FSD-MIX-CLIPS_data --data_type audio --setup mini

python Meta_train_FSC89.py -epochs_base 10 --trials 3 -episode_way 15 -episode_shot 1 \
-low_way 15 -low_shot 1 -lr_base 0.0001 -lrg 0.00001 -step 20 -gamma 0.5 -gpu 1 \
--metapath /SATA01/datasets/FSD-MIX-CLIPS-for_FSCIL/FSD_MIX_CLIPS.annotations_revised/FSC-89-meta \
--datapath /SATA01/datasets/FSD-MIX-CLIPS-for_FSCIL/FSD-MIX-CLIPS_data --data_type audio --setup mini