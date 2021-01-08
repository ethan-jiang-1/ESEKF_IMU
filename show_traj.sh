export MPLBACKEND=Qt5Agg
evo_config set plot_backend Qt5Agg
evo_traj tum ./data/traj_esekf_out.txt --ref ./data/traj_gt_out.txt -p