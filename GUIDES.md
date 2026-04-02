To load from an existing checkpoint:
- start the training making sure the config is the same (architecture, obs space, action space)
- stop the training after at least one checkpoint is saved
- copy your checkpoint in the nely created folder in the checkpoint_p0 folder and rename it as the latest one after deleting it
- start the training back again

python aerial_gym/rl_training/sample_factory/aerialgym_examples/train_aerialgym.py --env=matterport_vae_training_task --experiment=my_matterport_policy --train_dir=$(pwd)/train_dir --restart_behavior=resume --load_checkpoint_kind=latest