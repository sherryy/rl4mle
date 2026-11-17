# Reinforcement Learning for Machine Learning Engineering Agents

[Paper](https://arxiv.org/abs/2509.01684) · Sherry Yang, Joy He-Yueya, and Percy Liang (2025)

This repository focuses on reinforcement learning for machine learning engineering (MLE): a language model proposes a plan and Python code, the code produces a competition submission, and an MLEBench grader supplies a reward for PPO training. The training infrastructure builds on [veRL](https://github.com/volcengine/verl) and [TinyZero](https://github.com/Jiayi-Pan/TinyZero).

## Scope of this checkout

The retained workflow includes MLEBench task prompts, Parquet prompt preparation, distributed PPO training, parallel reward evaluation with Ray, and submission grading.

The paper introduces duration-aware policy gradients, environment instrumentation with a separate static model, and multi-step self-improvement. **Those extensions are not implemented in this checkout.** The current reward adapter uses raw grader scores and coarse submission bonuses; it does not implement the paper's partial-credit rewards or reverse scores for tasks where lower is better. The launcher is a starting configuration for the available PPO workflow, not a complete reproduction of the paper's experiments.

## Installation and environment

Use a Linux machine with NVIDIA GPUs and a CUDA environment compatible with the pinned vLLM version. Run repository commands from the repository root.

```bash
conda create -n rl-mle python=3.11
conda activate rl-mle

python -m pip install vllm==0.6.3
python -m pip install flash-attn --no-build-isolation
python -m pip install -r requirements.txt
python -m pip install -e .
```

Install [MLEBench](https://github.com/openai/mle-bench#setup) separately, including its Git LFS files, and make the `mlebench` command available to reward workers:

```bash
git clone https://github.com/openai/mle-bench.git ../mle-bench
git -C ../mle-bench lfs pull
python -m pip install -e ../mle-bench
```

Follow MLEBench's instructions for Kaggle credentials and preparing each competition, for example:

```bash
mlebench prepare -c random-acts-of-pizza
```

Competition data, model weights, and the packages needed by generated ML programs are external dependencies. Install the packages advertised in the selected task prompt in the execution environment. The launcher uses Weights & Biases by default; configure an account or override the logger as shown below.

### Local paths and execution resources

The reward execution integration retains assumptions from the original research environment. Configure these before launching training:

- Update dataset paths in `examples/data_preprocess/<competition_id>.txt` to the prepared competition's `public/` directory. The bundled prompts contain original machine-specific paths. Keep the instruction to write `./submission.csv` and the existing chat-template delimiters.
- Set `workspace_dir` consistently in `verl/trainer/main_ppo.py` and `verl/utils/reward_score/mlebench.py` to a dedicated writable directory. The training entry point clears that directory when it starts.
- Adjust the GPU assignment in `verl/utils/reward_score/interpreter.py`. It currently sets `CUDA_VISIBLE_DEVICES` to GPUs `8` through `15` for generated ML programs, separately from the launcher's training GPU count.

Generated programs execute inside Ray worker processes. The included interpreter provides neither a sandbox nor an enforced execution timeout. `data.timeout` limits how long the reward manager waits for a Ray result; it does not terminate the generated program.

## Prepare RL prompts

After updating the competition prompt's data paths, create the training and validation Parquet files:

```bash
export COMPETITION_ID=random-acts-of-pizza
export DATA_DIR="$HOME/data/mlebench/$COMPETITION_ID"

python examples/data_preprocess/mlebench.py \
  --competition_id "$COMPETITION_ID" \
  --local_dir "$DATA_DIR" \
  --train_size 32768 \
  --test_size 128
```

The preprocessor reads the matching `.txt` prompt next to the script and writes `train.parquet` and `test.parquet`. Each file repeats the task prompt with a distinct row index, so the policy can sample multiple solutions to the same competition. These are RL prompt splits; the actual competition training and test data come from MLEBench. The competition ID is stored in both `data_source` and `reward_model.ground_truth` for grading. An optional `--hdfs_dir` copies the generated files to HDFS.

## Train with PPO

With the execution environment configured and the prompt files prepared:

```bash
export N_GPUS=8
export BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
export ROLLOUT_TP_SIZE=8
export EXPERIMENT_NAME="$COMPETITION_ID-qwen2.5-3b-instruct"
export VLLM_ATTENTION_BACKEND=XFORMERS
export TIMEOUT=300

bash scripts/train_tiny_zero_mlebench.sh
```

`BASE_MODEL` can also point to a local Hugging Face model directory. Match `N_GPUS` and `ROLLOUT_TP_SIZE` to the training resources available. `run.sh` wraps the same launcher with environment-variable defaults.

The launcher accepts additional Hydra overrides. For example, to log to the console:

```bash
bash scripts/train_tiny_zero_mlebench.sh 'trainer.logger=[console]'
```

Configuration is defined in [`verl/trainer/config/ppo_trainer.yaml`](verl/trainer/config/ppo_trainer.yaml), with experiment overrides in [`scripts/train_tiny_zero_mlebench.sh`](scripts/train_tiny_zero_mlebench.sh).

| Setting | Purpose |
| --- | --- |
| `data.train_files`, `data.val_files` | Prepared RL prompt files. |
| `data.max_prompt_length`, `data.max_response_length` | Token limits; increase the prompt limit if a competition prompt exceeds it. |
| `data.timeout` | Wait limit for each training reward result. |
| `actor_rollout_ref.actor.optim.lr` | Policy learning rate. |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | Tensor parallelism for generation. |
| `trainer.save_freq`, `trainer.test_freq` | Checkpoint and validation intervals. |

Checkpoints are saved under `checkpoints/rl-for-mle/<experiment_name>/` by default, and the launcher writes `mlebench_train.log`. Validation uses the same MLEBench reward adapter on the configured validation prompts.

## Code map

| Path | Role |
| --- | --- |
| `examples/data_preprocess/mlebench.py` | Builds MLE RL prompt datasets. |
| `examples/data_preprocess/*.txt` | Competition instructions, formats, and data snippets. |
| `scripts/train_tiny_zero_mlebench.sh` | PPO launch configuration. |
| `verl/trainer/main_ppo.py` | Initializes workers and dispatches MLE rewards. |
| `verl/trainer/ppo/` | PPO training loop and optimization utilities. |
| `verl/utils/reward_score/mlebench.py` | Extracts generated code and invokes `mlebench grade-sample`. |
| `verl/utils/reward_score/interpreter.py` | Runs generated programs and captures execution output. |
| `verl/workers/`, `verl/single_controller/`, `verl/third_party/` | Shared distributed training and rollout infrastructure. |

## Citation

```bibtex
@article{yang2025reinforcement,
  title={Reinforcement Learning for Machine Learning Engineering Agents},
  author={Yang, Sherry and He-Yueya, Joy and Liang, Percy},
  journal={arXiv preprint arXiv:2509.01684},
  year={2025},
  url={https://arxiv.org/abs/2509.01684}
}
```

The inherited implementation retains its upstream copyright notices and [Apache 2.0 license](LICENSE). MLEBench supplies the competition preparation and grading tools; veRL and TinyZero supply the underlying RL infrastructure.
