
python Pre_train_Nsynth.py -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 --trials 1 \
-lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 100 -schedule Milestone -milestones 60 70 \
-gpu 0 -temperature 16 --metapath /SATA01/datasets/The_NSynth_Dataset \
--audiopath /SATA01/datasets/The_NSynth_Dataset

python Meta_train_Nsynth.py -epochs_base 10e0 --trials 100 -episode_way 15 -episode_shot 1 \
-low_way 15 -low_shot 1 -lr_base 0.0001 -lrg 0.00001 -step 20 -gamma 0.5 -gpu 1 --do_norm \
--metapath /SATA01/datasets/The_NSynth_Dataset --audiopath /SATA01/datasets/The_NSynth_Dataset \
--num_class 100 --base_class 55