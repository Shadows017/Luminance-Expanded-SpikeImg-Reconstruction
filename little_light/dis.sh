python train_lite_distill.py \
  --dataset_path ../../data/luminance_expanded_spike_x4k \
  --teacher_model checkpoint/2026-06-08_20_22_20/best_model_psnr:36.2755_epoch:44_.pth \
  --device cuda:0 \
  --student_init checkpoint/best_model_1000.pth \
  --etas 0.1 0.3 0.5 0.7 1.0 2.0 \
  --target_eta 0.5 \
  --use_light_code \
  --use_ldf_lite \
  --use_lsa_lite \
  --descriptor_dim 64 \
  --lambda_distill 0.2 \
  --grad_clip 1.0 \
  --amp

