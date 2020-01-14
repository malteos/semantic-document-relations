"""

Inputs:

- predefined config
- GPU

# check with nvidia-smi first

# Run from within tmux
python cli.py run ./output/cli/ 1 wiki.bert_base__joint__seq512

python cli.py run ./output/cli/ 2 wiki.bert_base__joint__seq256
...

"""

import json
import logging
import os
import pickle
import sys
from importlib import import_module

import fire

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def run(output_dir, gpu_id: int, config_name, **override_config):
    """

    :param output_dir:
    :param gpu_id: GPU (-1 == CPU)
    :param config_name: Predefined experiment config
    :param override_config: Use kwargs to override config variables, e.g., --foo__bar=1 (nested dict with __)
    :return:
    """
    output_dir = os.path.join(output_dir, config_name)

    logger.info(f'Starting... {config_name}')

    if os.path.exists(output_dir):
        logger.error(f'Output dir exist already: {output_dir}')
        sys.exit(1)

    # GPU
    if gpu_id < 0:
        logger.info('GPU is disabled')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # check with nvidia-smi

        import torch

        if not torch.cuda.is_available():
            logger.error('CUDA is not available!')
            sys.exit(1)

    # Predefined configs
    config_name = 'experiments.predefined.' + config_name
    try:
        package, module_name = config_name.rsplit('.', 1)
        module = import_module(package)
        config = getattr(module, module_name)

        assert isinstance(config, dict) == True
    except ValueError:
        logger.error(f'Cannot load experiment config from: {config_name}')
        sys.exit(1)

    # Override config
    from experiments.predefined import update
    from experiments.utils import unflatten

    if override_config:
        override_config = unflatten(override_config)

        logger.info(f'Override config with: {override_config}')
        config = update(config, override_config)

    from experiments import Experiment

    exp = Experiment(**config)

    exp.run(mode=2)

    # save
    os.makedirs(output_dir)
    exp.output_dir = output_dir

    with open(os.path.join(output_dir, 'experiment.pickle'), 'wb') as f:
        # json.dump(exp.to_dict(), f)
        pickle.dump(exp.to_dict(), f)

    with open(os.path.join(output_dir, 'reports.json'), 'w') as f:
        json.dump(exp.reports, f)

    exp.save()

    logger.info('Done')


def build_script(input_dir, output_dir, gpu_ids, missing_only=False, **override_config):
    """

    python cli.py build_script ./output/4fold ./output/4fold_results/ 0,1,2,3 --missing_only=1

    :param input_dir:
    :param output_dir:
    :param gpu_ids:
    :param missing_only:
    :param override_config:
    :return:
    """
    from experiments.utils import chunk

    configs = [
        'bert_base__joint__seq128',
        'bert_base__joint__seq256',
        'bert_base__joint__seq512',

        'bert_base__siamese__seq128__2d',
        'bert_base__siamese__seq128__3d',
        'bert_base__siamese__seq128__4d',

        'bert_base__siamese__seq256__2d',
        'bert_base__siamese__seq256__3d',
        'bert_base__siamese__seq256__4d',

        'bert_base__siamese__seq512__2d',
        'bert_base__siamese__seq512__3d',
        'bert_base__siamese__seq512__4d',

        'xlnet_base__joint__seq128',
        'xlnet_base__joint__seq256',
        'xlnet_base__joint__seq512',

        'xlnet_base__siamese__seq128__2d',
        'xlnet_base__siamese__seq128__3d',
        'xlnet_base__siamese__seq128__4d',

        'xlnet_base__siamese__seq256__2d',
        'xlnet_base__siamese__seq256__3d',
        'xlnet_base__siamese__seq256__4d',

        'xlnet_base__siamese__seq512__2d',
        'xlnet_base__siamese__seq512__3d',
        'xlnet_base__siamese__seq512__4d',
    ]


    # python cli.py run ./output/cli/ 2 wiki.bert_base__siamese__seq128__2d
    runs = []

    for k in sorted(os.listdir(input_dir)):
        for k_config in configs:
            runs.append((os.path.join(output_dir, k), os.path.join(input_dir, k), k_config))

    gpu_ids = list(gpu_ids) if isinstance(gpu_ids, list) else [gpu_ids]
    missing = 0

    for i, gpu_runs in enumerate(chunk(runs, len(gpu_ids))):
        gpu_id = gpu_ids[i]
        for out_dir, in_dir, cfg in gpu_runs:

            if not missing_only or not os.path.exists(out_dir):
                missing += 1
                print(f'python cli.py run {out_dir} {gpu_id} wiki.{cfg} --data_helper_params__train_dataframe_path={in_dir}/train.csv  --data_helper_params__test_dataframe_path={in_dir}/test.csv')
        #print(gpu_id)
        # pass
        print()

    print(f'# GPU cores: {len(gpu_ids)}')
    print(f'# Configs: {len(configs)}')
    print(f'# Runs: {len(runs)} (missing: {missing}')
    print(f'# Folds: {len(os.listdir(input_dir))}')


if __name__ == '__main__':
    fire.Fire()