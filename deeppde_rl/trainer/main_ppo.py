import os
import hydra
import ray


def get_reward_function(config):
    reward_fn_config = config.get("reward_function") or {}
    file_path = reward_fn_config.get("path")
    print(file_path)
    if not file_path:
        raise ValueError("Path to reward function not found")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    import importlib

    function_name = reward_fn_config.get("name")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    reward_fn = getattr(module, function_name)

    return reward_fn
    

@hydra.main(config_path='config', config_name='ppo', version_base=None)
def main(config):
    run_ppo(config)

def run_ppo(config):
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={
            'env_vars': {
                # 'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'VLLM_LOGGING_LEVEL': 'WARN'
            }
        })

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


class TaskRunner:
    def run(self, config):
        from pprint import pprint
        from omegaconf import OmegaConf
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        local_path = config.actor_rollout_ref.model.path

        # TODO: instantiate tokenizer

        # TODO: add other strategy
        # define worker classes
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            # from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
            # from verl.single_controller.ray import RayWorkerGroup
            # ray_worker_group_cls = RayWorkerGroup

        else:
            raise ValueError(f"Invalid strategy: {config.actor_rollout_ref.actor.strategy}")



