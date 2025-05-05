
## step1 启动远程的reward模型服务
1. 在非训练机器上（或者任何机器上），启动reward服务。
2. 脚本

```bash

cd reward_part

python reward_api.py

```

3. 留意reward模型对应的服务器ip和端口


4. 在训练模型前期，一定要debug奖励模型。看看我之前踩的坑`reward_part/client.ipynb`

## step2 开始训练
1. 按照verl的要求，整理好数据。
2. 参考`train_grpo_cr.sh`脚本，将里面的相关参数进行修改。
- 2.1 `WANDB_API_KEY` 、`data_dir` 、 `model_path` 、`save_model_checkpoint`
- 2.2 `reward_model.reward_api`: 填写刚才启动的reward模型服务

```bash

# 官方文档链接，详细介绍了所有参数: https://verl.readthedocs.io/en/latest/examples/config.html#config-explain-page
# 这个是我自定义的参数，自定义远程的奖励模型服务: reward_model.reward_api=http://10.xxx.0.xxx:6009/get_reward2 \

export CUDA_VISIBLE_DEVICES="4,5,6,7"

set -x

# export RAY_DEBUG=1
export WANDB_API_KEY=xxxxxx
ray stop
ray start --head --node-ip-address=0.0.0.0 --port=6378 --dashboard-host=0.0.0.0 --dashboard-port=8265 --ray-debugger-external --num-gpus 4


# 数据路径
data_dir=/xxx/train_rlhf_on_cluster/data/clean_huipeng_data
# 模型路径
model_path=/xxxxx/weights/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# 训练任务wandb
cur_task=verl_custom_test
save_model_checkpoint=/xxx/train_rlhf_on_cluster/train_models/$cur_task

echo "在主节点上启动训练..."
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$data_dir/train.parquet \
    data.val_files=$data_dir/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.truncation=right \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_api=http://10.xxx.0.xxx:6009/get_reward2 \ # 注意这里的api
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_custom' \
    trainer.experiment_name=$cur_task \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.default_local_dir=$save_model_checkpoint \
    trainer.test_freq=5 \
    trainer.total_epochs=1 $@



```