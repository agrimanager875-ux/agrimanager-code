"""Entry point for AgriManager RL training with env metrics logging.

Uses AgriTrainer (standalone PPO trainer) to log per-episode validation data
and env trajectory metrics to the configured experiment tracker. Follows verl's
main_ppo.py pattern with a custom TaskRunner.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf, open_dict

from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device

from agrimanager.adapter.trainer.trainer import AgriTrainer
from agrimanager.adapter.trainer.validation_sets import (
    EnvConfigOverrideDataset,
    create_named_validation_dataset,
    flatten_val_set_files,
    normalize_env_config_overrides,
    normalize_val_sets,
)


class AgriTaskRunner(TaskRunner):
    """TaskRunner that uses AgriTrainer instead of RayPPOTrainer."""

    def run(self, config):
        from pprint import pprint

        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        print(f"AgriTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        data_config = config.data
        named_val_sets = normalize_val_sets(data_config.get("val_sets", None))
        env_config_overrides = normalize_env_config_overrides(
            data_config.get("env_config_overrides", None)
        )
        val_files = data_config.get("val_files", None)
        has_val_data = bool(named_val_sets) or bool(val_files)
        test_freq = config.trainer.get("test_freq", 0) or 0
        validation_requested = (
            bool(config.trainer.get("val_before_train", True))
            or bool(test_freq > 0)
            or bool(config.trainer.get("val_only", False))
        )
        validation_enabled = validation_requested and has_val_data
        if validation_requested and not has_val_data:
            raise ValueError(
                "Validation is requested but no validation data is configured. "
                "Set data.val_files or data.val_sets, or disable validation with "
                "trainer.val_before_train=False and trainer.test_freq=0."
            )

        if validation_enabled and named_val_sets:
            with open_dict(data_config):
                data_config.val_files = flatten_val_set_files(named_val_sets)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = None
        if validation_enabled:
            val_reward_fn = load_reward_manager(
                config,
                tokenizer,
                num_examine=1,
                **config.reward_model.get("reward_kwargs", {}),
            )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        train_dataset = create_rl_dataset(
            data_config.train_files,
            data_config,
            tokenizer,
            processor,
            is_train=True,
            max_samples=data_config.get("train_max_samples", -1),
        )
        if env_config_overrides:
            train_dataset = EnvConfigOverrideDataset(train_dataset, env_config_overrides)
        val_dataset = None
        if validation_enabled and named_val_sets:
            val_dataset = create_named_validation_dataset(
                named_val_sets,
                data_config,
                tokenizer,
                processor,
                create_rl_dataset,
                max_samples=data_config.get("val_max_samples", -1),
            )
        elif validation_enabled:
            val_dataset = create_rl_dataset(
                data_config.val_files,
                data_config,
                tokenizer,
                processor,
                is_train=False,
                max_samples=data_config.get("val_max_samples", -1),
            )
        if val_dataset is not None and env_config_overrides:
            val_dataset = EnvConfigOverrideDataset(val_dataset, env_config_overrides)
        train_sampler = create_rl_sampler(data_config, train_dataset)

        # Use AgriTrainer instead of RayPPOTrainer
        trainer = AgriTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    task_runner_class = ray.remote(num_cpus=1)(AgriTaskRunner)
    run_ppo(config, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
