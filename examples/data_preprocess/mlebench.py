"""Prepare repeated MLEBench competition prompts for RL training and evaluation."""

import argparse
from pathlib import Path

from datasets import Dataset
from verl.utils.hdfs_io import copy, makedirs


def make_dataset(prompt, competition_id, split, size):
    """Create prompt records; competition data is prepared separately with MLEBench."""
    return Dataset.from_list([
        {
            "data_source": competition_id,
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "machine_learning_engineering",
            "reward_model": {
                "style": "rule",
                "ground_truth": competition_id,
            },
            "extra_info": {"split": split, "index": index},
        }
        for index in range(size)
    ])


if __name__ == '__main__':
    prompt_dir = Path(__file__).resolve().parent
    competition_ids = sorted(path.stem for path in prompt_dir.glob('*.txt'))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local_dir', default='~/data/mlebench')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_size', type=int, default=32768)
    parser.add_argument('--test_size', type=int, default=128)
    parser.add_argument('--competition_id', choices=competition_ids, default='spaceship-titanic')
    args = parser.parse_args()

    if args.train_size <= 0 or args.test_size <= 0:
        parser.error('--train_size and --test_size must be positive')

    prompt = (prompt_dir / f'{args.competition_id}.txt').read_text(encoding='utf-8')
    local_dir = Path(args.local_dir).expanduser()
    local_dir.mkdir(parents=True, exist_ok=True)

    for split, size in [('train', args.train_size), ('test', args.test_size)]:
        dataset = make_dataset(prompt, args.competition_id, split, size)
        dataset.to_parquet(str(local_dir / f'{split}.parquet'))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=str(local_dir), dst=args.hdfs_dir)
