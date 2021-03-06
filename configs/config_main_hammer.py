DEFAULT_CONFIG = dict(
  bc_epoch= 20,
  expert_policy_folder= "hand_hammer_expert",
  env_id= "mjrl_hammer-v0",
  val_traj_per_file= 5,
  has_robot_info= True,
  gen_traj_dagger_ep= 50,
  trainer_epochs= 10,
  seed= 1000,
  id_post= "public_final_dagger",
  num_files_train= 1,
  use_late_fusion= True,
  num_files_val= 1,
  camera_name= "vil_camera",
  dagger_epoch= 20,
  beta_decay= 0.2,
  viz_policy_folder_dagger= "dagger_hand_hammer_viz_policy",
  eval_num_traj= 100,
  device_id= 0,
  delta= 0.01,
  env_name= "hand_hammer",
  sliding_window= 80,
  use_tactile= True,
  batch_size_viz_pol= 128,
  use_cuda= True,
  lr= 0.0003,
  horizon_il= 150,
  train_expert= False,
  num_traj_expert= 20,
  traj_budget_expert= 12500,
  train_traj_per_file= 50,
  beta_start= 1
)
