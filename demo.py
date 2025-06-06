import configparser
import argparse
import shutil
import glob
import uuid
import ast
import os

import torch
import pufferlib
import pufferlib.utils
import pufferlib.vector
import pufferlib.cleanrl

from rich_argparse import RichHelpFormatter
from rich.console import Console
from rich.traceback import install
install(show_locals=False) # Rich tracebacks

import signal # Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

import clean_pufferl


import re
import pickle
from kcore_functions import *

def make_policy(env, policy_cls, rnn_cls, args):
    policy = policy_cls(env, **args['policy'])
    if rnn_cls is not None:
        policy = rnn_cls(env, policy, **args['rnn'])
        policy = pufferlib.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.cleanrl.Policy(policy)

    return policy.to(args['train']['device'])

def make_policy_kcore(env, policy_cls, rnn_cls, args):
    
    if not args.get('eval_model_path'):
        raise ValueError("You must provide a path to a pretrained model in 'eval_model_path'")
    
    print(f"Loading model from {args['eval_model_path']}")
    model = torch.load(args['eval_model_path'], map_location=args['train']['device'])
    
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Loaded object is not a full model (torch.nn.Module). Did you save with torch.save(model)?")
    
    return model.to(args['train']['device'])

def init_wandb(args, name, id=None, resume=True):
    import wandb
    wandb.init(
        id=id or wandb.util.generate_id(),
        project=args['wandb_project'],
        group=args['wandb_group'],
        allow_val_change=True,
        save_code=True,
        resume=resume,
        config=args,
        name=name,
    )
    return wandb

def sweep(args, env_name, make_env, policy_cls, rnn_cls):
    import wandb
    sweep_id = wandb.sweep(sweep=args['sweep'], project=args['wandb_project'])

    def main():
        try:
            wandb = init_wandb(args, env_name, id=args['exp_id'])
            args['train'].update(wandb.config.train)
            train(args, make_env, policy_cls, rnn_cls, wandb)
        except Exception as e:
            Console().print_exception()

    wandb.agent(sweep_id, main, count=100)

### CARBS Sweeps
def sweep_carbs(args, env_name, make_env, policy_cls, rnn_cls):
    import numpy as np
    import sys

    from math import log, ceil, floor

    from carbs import CARBS
    from carbs import CARBSParams
    from carbs import LinearSpace
    from carbs import LogSpace
    from carbs import LogitSpace
    from carbs import ObservationInParam
    from carbs import ParamDictType
    from carbs import Param

    def closest_power(x):
        possible_results = floor(log(x, 2)), ceil(log(x, 2))
        return int(2**min(possible_results, key= lambda z: abs(x-2**z)))

    def carbs_param(group, name, space, wandb_params, mmin=None, mmax=None,
            search_center=None, is_integer=False, rounding_factor=1, scale=1):
        wandb_param = wandb_params[group]['parameters'][name]
        if 'values' in wandb_param:
            values = wandb_param['values']
            mmin = min(values)
            mmax = max(values)

        if mmin is None:
            mmin = float(wandb_param['min'])
        if mmax is None:
            mmax = float(wandb_param['max'])

        if space == 'log':
            Space = LogSpace
            if search_center is None:
                search_center = 2**(np.log2(mmin) + np.log2(mmax)/2)
        elif space == 'linear':
            Space = LinearSpace
            if search_center is None:
                search_center = (mmin + mmax)/2
        elif space == 'logit':
            Space = LogitSpace
            assert mmin == 0
            assert mmax == 1
            assert search_center is not None
        else:
            raise ValueError(f'Invalid CARBS space: {space} (log/linear)')

        return Param(
            name=f'{group}/{name}',
            space=Space(
                min=mmin,
                max=mmax,
                is_integer=is_integer,
                rounding_factor=rounding_factor,
                scale=scale,
            ),
            search_center=search_center,
        )

    if not os.path.exists('checkpoints'):
        os.system('mkdir checkpoints')

    import wandb
    sweep_id = wandb.sweep(
        sweep=args['sweep'],
        project="carbs",
    )
    target_metric = args['sweep']['metric']['name'].split('/')[-1]
    sweep_parameters = args['sweep']['parameters']
    #wandb_env_params = sweep_parameters['env']['parameters']
    #wandb_policy_params = sweep_parameters['policy']['parameters']

    # Must be hardcoded and match wandb sweep space for now
    param_spaces = []
    if 'total_timesteps' in sweep_parameters['train']['parameters']:
        time_param = sweep_parameters['train']['parameters']['total_timesteps']
        min_timesteps = time_param['min']
        param_spaces.append(carbs_param('train', 'total_timesteps', 'log', sweep_parameters,
            search_center=min_timesteps, is_integer=True))

    batch_param = sweep_parameters['train']['parameters']['batch_size']
    default_batch = (batch_param['max'] - batch_param['min']) // 2

    minibatch_param = sweep_parameters['train']['parameters']['minibatch_size']
    default_minibatch = (minibatch_param['max'] - minibatch_param['min']) // 2

    if 'env' in sweep_parameters:
        env_params = sweep_parameters['env']['parameters']

        # MOBA
        if 'reward_death' in env_params:
            param_spaces.append(carbs_param('env', 'reward_death',
                'linear', sweep_parameters, search_center=-0.42))
        if 'reward_xp' in env_params:
            param_spaces.append(carbs_param('env', 'reward_xp',
                'linear', sweep_parameters, search_center=0.015, scale=0.05))
        if 'reward_distance' in env_params:
            param_spaces.append(carbs_param('env', 'reward_distance',
                'linear', sweep_parameters, search_center=0.15, scale=0.5))
        if 'reward_tower' in env_params:
            param_spaces.append(carbs_param('env', 'reward_tower',
                'linear', sweep_parameters, search_center=4.0))

        # Atari
        if 'frameskip' in env_params:
            param_spaces.append(carbs_param('env', 'frameskip',
                'linear', sweep_parameters, search_center=4, is_integer=True))
        if 'repeat_action_probability' in env_params:
            param_spaces.append(carbs_param('env', 'repeat_action_probability',
                'logit', sweep_parameters, search_center=0.25))

    param_spaces += [
        #carbs_param('cnn_channels', 'linear', wandb_policy_params, search_center=32, is_integer=True),
        #carbs_param('hidden_size', 'linear', wandb_policy_params, search_center=128, is_integer=True),
        #carbs_param('vision', 'linear', search_center=5, is_integer=True),
        carbs_param('train', 'learning_rate', 'log', sweep_parameters, search_center=1e-3),
        carbs_param('train', 'gamma', 'logit', sweep_parameters, search_center=0.95),
        carbs_param('train', 'gae_lambda', 'logit', sweep_parameters, search_center=0.90),
        carbs_param('train', 'update_epochs', 'linear', sweep_parameters,
            search_center=1, scale=3, is_integer=True),
        carbs_param('train', 'clip_coef', 'logit', sweep_parameters, search_center=0.1),
        carbs_param('train', 'vf_coef', 'logit', sweep_parameters, search_center=0.5),
        carbs_param('train', 'vf_clip_coef', 'logit', sweep_parameters, search_center=0.1),
        carbs_param('train', 'max_grad_norm', 'linear', sweep_parameters, search_center=0.5),
        carbs_param('train', 'ent_coef', 'log', sweep_parameters, search_center=0.01),
        carbs_param('train', 'batch_size', 'log', sweep_parameters,
            search_center=default_batch, is_integer=True),
        carbs_param('train', 'minibatch_size', 'log', sweep_parameters,
            search_center=default_minibatch, is_integer=True),
        carbs_param('train', 'bptt_horizon', 'log', sweep_parameters,
            search_center=16, is_integer=True),
    ]

    carbs_params = CARBSParams(
        better_direction_sign=1,
        is_wandb_logging_enabled=False,
        resample_frequency=5,
        num_random_samples=len(param_spaces),
        max_suggestion_cost=args['max_suggestion_cost'],
        is_saved_on_every_observation=False,
    )
    carbs = CARBS(carbs_params, param_spaces)

    # GPUDrive doesn't let you reinit the vecenv, so we have to cache it
    cache_vecenv = args['env_name'] == 'gpudrive'

    elos = {'model_random.pt': 1000}
    vecenv = {'vecenv': None} # can't reassign otherwise
    shutil.rmtree('moba_elo', ignore_errors=True)
    os.mkdir('moba_elo')
    import time, torch
    def main():
        print('Vecenv:', vecenv)
        # set torch and pytorch seeds to current time
        np.random.seed(int(time.time()))
        torch.manual_seed(int(time.time()))
        wandb = init_wandb(args, env_name, id=args['exp_id'])
        wandb.config.__dict__['_locked'] = {}
        orig_suggestion = carbs.suggest().suggestion
        suggestion = orig_suggestion.copy()
        print('Suggestion:', suggestion)
        #cnn_channels = suggestion.pop('cnn_channels')
        #hidden_size = suggestion.pop('hidden_size')
        #vision = suggestion.pop('vision')
        #wandb.config.env['vision'] = vision
        #wandb.config.policy['cnn_channels'] = cnn_channels
        #wandb.config.policy['hidden_size'] = hidden_size
        train_suggestion = {k.split('/')[1]: v for k, v in suggestion.items() if k.startswith('train/')}
        env_suggestion = {k.split('/')[1]: v for k, v in suggestion.items() if k.startswith('env/')}
        args['train'].update(train_suggestion)
        args['train']['batch_size'] = closest_power(
            train_suggestion['batch_size'])
        args['train']['minibatch_size'] = closest_power(
            train_suggestion['minibatch_size'])
        args['train']['bptt_horizon'] = closest_power(
            train_suggestion['bptt_horizon'])

        args['env'].update(env_suggestion)
        args['track'] = True
        wandb.config.update({'train': args['train']}, allow_val_change=True)
        wandb.config.update({'env': args['env']}, allow_val_change=True)

        #args.env.__dict__['vision'] = vision
        #args['policy']['cnn_channels'] = cnn_channels
        #args['policy']['hidden_size'] = hidden_size
        #args['rnn']['input_size'] = hidden_size
        #args['rnn']['hidden_size'] = hidden_size
        print(wandb.config.train)
        print(wandb.config.env)
        print(wandb.config.policy)
        try:
            stats, uptime, new_elos, vecenv['vecenv'] = train(args, make_env, policy_cls, rnn_cls,
                wandb, elos=elos, vecenv=vecenv['vecenv'] if cache_vecenv else None)
            elos.update(new_elos)
        except Exception as e:
            import traceback
            traceback.print_exc()
        else:
            observed_value = stats[target_metric]
            print('Observed value:', observed_value)
            print('Uptime:', uptime)

            carbs.observe(
                ObservationInParam(
                    input=orig_suggestion,
                    output=observed_value,
                    cost=uptime,
                )
            )

    wandb.agent(sweep_id, main, count=500)

def train(args, make_env, policy_cls, rnn_cls, wandb,
        eval_frac=0.1, elos={'model_random.pt': 1000}, vecenv=None, subprocess=False, queue=None):
    if subprocess:
        from multiprocessing import Process, Queue
        queue = Queue()
        p = Process(target=train, args=(args, make_env, policy_cls, rnn_cls, wandb,
            eval_frac, elos, False, queue))
        p.start()
        p.join()
        stats, uptime, elos = queue.get()

    if args['vec'] == 'serial':
        vec = pufferlib.vector.Serial
    elif args['vec'] == 'multiprocessing':
        vec = pufferlib.vector.Multiprocessing
    elif args['vec'] == 'ray':
        vec = pufferlib.vector.Ray
    elif args['vec'] == 'native':
        vec = pufferlib.environment.PufferEnv
    else:
        raise ValueError(f'Invalid --vec (serial/multiprocessing/ray/native).')

    if vecenv is None:
        vecenv = pufferlib.vector.make(
            make_env,
            env_kwargs=args['env'],
            num_envs=args['train']['num_envs'],
            num_workers=args['train']['num_workers'],
            batch_size=args['train']['env_batch_size'],
            zero_copy=args['train']['zero_copy'],
            overwork=args['vec_overwork'],
            backend=vec,
        )

        
    # DECIDE WHICH POLICY FUNCTION TO USE. USED THE MODIFIED IF COLLAPSE!=0. Matteo Serafino
    if args['collapse'] != 0:
        policy = make_policy_kcore(vecenv.driver_env, policy_cls, rnn_cls, args)
    else:
        policy = make_policy(vecenv.driver_env, policy_cls, rnn_cls, args)

    '''
    if env_name == 'moba':
        import torch
        os.makedirs('moba_elo', exist_ok=True)
        torch.save(policy, os.path.join('moba_elo', 'model_random.pt'))
    '''
    train_config = pufferlib.namespace(**args['train'], env=env_name,
        exp_id=args['exp_id'] or env_name + '-' + str(uuid.uuid4())[:8])
    data = clean_pufferl.create(train_config, vecenv, policy, wandb=wandb)

    while data.global_step < train_config.total_timesteps:
        clean_pufferl.evaluate(data)
        clean_pufferl.train(data)
    
    uptime = data.profile.uptime
    steps_evaluated = 0
    steps_to_eval = int(args['train']['total_timesteps'] * eval_frac)
    batch_size = args['train']['batch_size']
    while steps_evaluated < steps_to_eval:
        stats, _ = clean_pufferl.evaluate(data)
        steps_evaluated += batch_size

    clean_pufferl.mean_and_log(data)

    '''
    if env_name == 'moba':
        exp_n = len(elos)
        model_name = f'model_{exp_n}.pt'
        torch.save(policy, os.path.join('moba_elo', model_name))
        from evaluate_elos import calc_elo
        elos = calc_elo(model_name, 'moba_elo', elos)
        stats['elo'] = elos[model_name]
        if wandb is not None:
            wandb.log({'environment/elo': elos[model_name]})
    '''

    clean_pufferl.close(data)
    if queue is not None:
        queue.put((stats, uptime, elos))

    return stats, uptime, elos, vecenv


# Collapse function. Matteo Serafino
def collapse_model(pt_file, threshold):
    """
    Loads a model and collapses its Linear and LSTM layers based on fiber clustering at the given threshold.

    Args:
        pt_file (str): Path to the model checkpoint (.pt file).
        threshold (float): Fiber clustering threshold (typically between 0 and 2).

    Returns:
        Tuple[torch.nn.Module, dict]: Collapsed model and corresponding fiber clustering data.
    """
    
    model = torch.load(pt_file, weights_only=False, map_location='cpu')
    fibers = get_fibers_vs_time([pt_file], threshold)

    clusters_linear = torch.tensor(fibers[(3, 'Linear(in_features=3456, out_features=512, bias=True)')][-1])
    clusters_lstm = torch.tensor(fibers[(4, 'LSTM(512, 512)')][-1])

    model_collapse_linear = collapse_linear_layer(model, clusters_linear)
    new_linear_clusters = torch.arange(model_collapse_linear.policy.policy.network[7].weight.shape[0])
    model_collapse_lstm = collapse_lstm(model_collapse_linear, new_linear_clusters, clusters_lstm)

    return model_collapse_lstm, fibers


# Run Collapse function. Matteo Serafino
def run_model_collapse(args, episode=50, default_folder='experiments'):
    """
    Loads a model checkpoint, applies fiber-based layer collapsing, and saves the compressed model and fibers.

    Args:
        args (dict): Parsed arguments containing the 'collapse' threshold.
        episode (int, optional): Target episode number to load (default is 50).
        default_folder (str, optional): Root folder to search for .pt files (default is 'experiments').

    Returns:
        str or None: Path to the collapsed model if collapsing is performed; otherwise, None.
    """
    threshold_fiber = args['collapse']
    if threshold_fiber == 0:
        print("[collapse] Collapse is disabled (threshold = 0).")
        return None

    # Format episode like model_000050.pt
    padded_episode = f"{episode:06d}"
    expected_filename = f"model_{padded_episode}.pt"

    # Find all .pt files in experiments
    pt_files = sorted(glob.glob(os.path.join(default_folder, '**', 'model_*.pt'), recursive=True))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found under '{default_folder}'")

    # Try to find exact match
    target_file = None
    for f in pt_files:
        if f.endswith(expected_filename):
            target_file = f
            break

    if not target_file:
        print(f"[collapse] Default file '{expected_filename}' not found. Searching for closest.")
        
        def extract_episode_num(path):
            match = re.search(r'model_(\d{6})\.pt$', path)
            return int(match.group(1)) if match else float('inf')

        pt_files_with_episodes = [(f, extract_episode_num(f)) for f in pt_files]
        pt_files_with_episodes = [(f, ep) for f, ep in pt_files_with_episodes if ep != float('inf')]
        closest_file, closest_episode = min(pt_files_with_episodes, key=lambda x: abs(x[1] - episode))
        
        print(f"[collapse] Using closest checkpoint: {closest_file}")
        target_file = closest_file
        episode_used = f"{closest_episode:06d}"
    else:
        episode_used = padded_episode

    print(f"[collapse] Collapsing model: {target_file} with threshold {threshold_fiber}")
    model_collapse_lstm, fibers = collapse_model(target_file, threshold_fiber)

    # Format threshold as two-digit int (e.g., 0.2 → 02, 1.2 → 12)
    thr_id = f"{int(threshold_fiber * 10):02d}"
    output_folder = os.path.join(os.path.dirname(target_file), f'collapsed_model_thr_{thr_id}_{episode_used}')
    os.makedirs(output_folder, exist_ok=True)

    # Save fibers
    fibers_out_path = os.path.join(output_folder, 'fibers.pkl')
    with open(fibers_out_path, 'wb') as f:
        pickle.dump(fibers, f)

    # Save collapsed model
    model_out_path = os.path.join(output_folder, f'model_{episode_used}_collapsed.pt')
    torch.save(model_collapse_lstm, model_out_path)

    print(f"[collapse] Saved collapsed model to: {model_out_path}")

    # Update args with new model path and data directory
    args['train']['data_dir'] = output_folder
    args['eval_model_path'] = model_out_path

    # Compute adjusted total_timesteps for exactly 300 training steps
    try:
        batch_size = args['train']['batch_size']
        total_target_steps = 300
        episode_number = int(episode_used)
        remaining_steps = total_target_steps - episode_number
        new_total_timesteps = remaining_steps * batch_size
        args['train']['total_timesteps'] = new_total_timesteps
        print(f"[collapse] Updated total_timesteps to {new_total_timesteps} for {remaining_steps} steps from episode {episode_number}.")
    except Exception as e:
        print(f"[collapse] Warning: Could not update total_timesteps. {e}")

    return model_out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f':blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]'
        ' demo options. Shows valid args for your env and policy',
        formatter_class=RichHelpFormatter, add_help=False)
    parser.add_argument('--env', '--environment', type=str,
        default='puffer_squared', help='Name of specific environment to run')
    parser.add_argument('--mode', type=str, default='train',
        choices='train eval evaluate sweep sweep-carbs autotune profile'.split())
    parser.add_argument('--vec-overwork', action='store_true',
        help='Allow vectorization to use >1 worker/core. Not recommended.')
    parser.add_argument('--eval-model-path', type=str, default=None,
        help='Path to a pretrained checkpoint')
    parser.add_argument('--baseline', action='store_true',
        help='Load pretrained model from WandB if available')
    parser.add_argument('--render-mode', type=str, default='auto',
        choices=['auto', 'human', 'ansi', 'rgb_array', 'raylib', 'None'])
    parser.add_argument('--exp-id', '--exp-name', type=str,
        default=None, help='Resume from experiment')
    parser.add_argument('--track', action='store_true', help='Track on WandB')
    parser.add_argument('--wandb-project', type=str, default='pufferlib')
    parser.add_argument('--wandb-group', type=str, default='debug')


    # NEW ARGUMENT: FIBER THRESHOLD. Matteo Serafino
    parser.add_argument('--collapse', type=float, default=0.0,
        help='Collapse level (float between 0 and 2, 0 = disabled)')

    args = parser.parse_known_args()[0]
        
    # Manual range check. Matteo Serafino
    if not (0.0 <= args.collapse <= 2.0):
        raise ValueError(f"--collapse must be between 0 and 2 (got {args.collapse})")

    file_paths = glob.glob('config/**/*.ini', recursive=True)
    for path in file_paths:
        p = configparser.ConfigParser()
        p.read('config/default.ini')

        subconfig = os.path.join(*path.split('/')[:-1] + ['default.ini'])
        if subconfig in file_paths:
            p.read(subconfig)

        p.read(path)
        if args.env in p['base']['env_name'].split():
            break
    else:
        raise Exception('No config for env_name {}'.format(args.env))

    for section in p.sections():
        for key in p[section]:
            if section == 'base':
                argparse_key = f'--{key}'.replace('_', '-')
            else:
                argparse_key = f'--{section}.{key}'.replace('_', '-')
            parser.add_argument(argparse_key, default=p[section][key])

    # Late add help so you get a dynamic menu based on the env
    parser.add_argument('-h', '--help', default=argparse.SUPPRESS,
        action='help', help='Show this help message and exit')

    parsed = parser.parse_args().__dict__
    args = {'env': {}, 'policy': {}, 'rnn': {}}
    env_name = parsed.pop('env')
    for key, value in parsed.items():
        next = args
        for subkey in key.split('.'):
            if subkey not in next:
                next[subkey] = {}
            prev = next
            next = next[subkey]
        try:
            prev[subkey] = ast.literal_eval(value)
        except:
            prev[subkey] = value

    package = args['package']
    module_name = f'pufferlib.environments.{package}'
    if package == 'ocean':
        module_name = 'pufferlib.ocean'

    import importlib
    env_module = importlib.import_module(module_name)

    make_env = env_module.env_creator(env_name)
    policy_cls = getattr(env_module.torch, args['policy_name'])
    
    rnn_name = args['rnn_name']
    rnn_cls = None
    if rnn_name is not None:
        rnn_cls = getattr(env_module.torch, args['rnn_name'])

    if args['baseline']:
        assert args['mode'] in ('train', 'eval', 'evaluate')
        args['track'] = True
        version = '.'.join(pufferlib.__version__.split('.')[:2])
        args['exp_id'] = f'puf-{version}-{env_name}'
        args['wandb_group'] = f'puf-{version}-baseline'
        shutil.rmtree(f'experiments/{args["exp_id"]}', ignore_errors=True)
        run = init_wandb(args, args['exp_id'], resume=False)
        if args['mode'] in ('eval', 'evaluate'):
            model_name = f'puf-{version}-{env_name}_model:latest'
            artifact = run.use_artifact(model_name)
            data_dir = artifact.download()
            model_file = max(os.listdir(data_dir))
            args['eval_model_path'] = os.path.join(data_dir, model_file)
    if args['mode'] == 'train':
        wandb = None
        if args['track']:
            wandb = init_wandb(args, env_name, id=args['exp_id'])

        # Collapse the model before training. Matteo Serafino
        if args['collapse']!=0:
            run_model_collapse(args,episode = 50)
            #exit(0)
        train(args, make_env, policy_cls, rnn_cls, wandb=wandb)
    elif args['mode'] in ('eval', 'evaluate'):
        vec = pufferlib.vector.Serial
        if args['vec'] == 'native':
            vec = pufferlib.environment.PufferEnv

        clean_pufferl.rollout(
            make_env,
            args['env'],
            policy_cls=policy_cls,
            rnn_cls=rnn_cls,
            agent_creator=make_policy,
            agent_kwargs=args,
            backend=vec,
            model_path=args['eval_model_path'],
            render_mode=args['render_mode'],
            device=args['train']['device'],
        )
    elif args['mode'] == 'sweep':
        args['track'] = True
        sweep(args, env_name, make_env, policy_cls, rnn_cls)
    elif args['mode'] == 'sweep-carbs':
        sweep_carbs(args, env_name, make_env, policy_cls, rnn_cls)
    elif args['mode'] == 'autotune':
        pufferlib.vector.autotune(make_env, batch_size=args['train']['env_batch_size'])
    elif args['mode'] == 'profile':
        import cProfile
        cProfile.run('train(args, make_env, policy_cls, rnn_cls, wandb=None)', 'stats.profile')
        import pstats
        from pstats import SortKey
        p = pstats.Stats('stats.profile')
        p.sort_stats(SortKey.TIME).print_stats(10)
