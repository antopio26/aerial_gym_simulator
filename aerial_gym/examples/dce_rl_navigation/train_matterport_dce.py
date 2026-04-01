import sys
import isaacgym  # noqa: F401 — must be imported before torch
import aerial_gym.task  # noqa: F401 — registers matterport_vae_training_task in task_registry

from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym import main

if __name__ == "__main__":
    sys.exit(main())
