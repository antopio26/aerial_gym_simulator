import argparse
import time

import numpy as np

from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.logging import CustomLogger
import torch


logger = CustomLogger(__name__)


def _build_env(rendering_benchmark: bool):
    if rendering_benchmark:
        env_manager = SimBuilder().build_env(
            sim_name="base_sim",
            env_name="env_with_obstacles",
            robot_name="base_quadrotor_with_camera",
            controller_name="lee_velocity_control",
            args=None,
            device="cuda:0",
            num_envs=16,
            headless=True,
            use_warp=True,
        )
        if not env_manager.robot_manager.robot.cfg.sensor_config.enable_camera:
            raise RuntimeError("Camera is disabled; rendering benchmark requires camera sensors.")
        return env_manager

    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="empty_env",
        robot_name="base_quadrotor",
        controller_name="no_control",
        args=None,
        device="cuda:0",
        num_envs=256,
        headless=True,
        use_warp=True,
    )
    if env_manager.robot_manager.robot.cfg.sensor_config.enable_camera:
        logger.warning(
            "Camera is enabled for physics benchmark; this will reduce benchmark throughput."
        )
    return env_manager


def main():
    parser = argparse.ArgumentParser(description="Simple simulator benchmark")
    parser.add_argument(
        "--rendering",
        action="store_true",
        help="Run rendering benchmark (camera-enabled) instead of physics-only benchmark.",
    )
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    args = parser.parse_args()

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logger.warning(
        "This script benchmarks environment throughput. Use --rendering for camera benchmark."
    )

    env_manager = _build_env(rendering_benchmark=args.rendering)
    actions = torch.zeros((env_manager.num_envs, 4), device="cuda:0")
    env_manager.reset()

    start = time.time()
    elapsed_steps = 0
    with torch.no_grad():
        for i in range(args.steps):
            if i == args.warmup_steps:
                start = time.time()
                elapsed_steps = 0
            env_manager.step(actions=actions)
            if args.rendering:
                env_manager.render(render_components="sensors")
            elapsed_steps += 1

            if i % 50 == 0 and i >= args.warmup_steps:
                elapsed = max(time.time() - start, 1.0e-9)
                logger.warning(
                    "steps=%d elapsed=%.3fs FPS=%.2f RTF=%.2f",
                    elapsed_steps,
                    elapsed,
                    elapsed_steps * env_manager.num_envs / elapsed,
                    elapsed_steps * env_manager.num_envs * env_manager.sim_config.sim.dt / elapsed,
                )


if __name__ == "__main__":
    main()
