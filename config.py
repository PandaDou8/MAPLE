import yaml
import sys
import os
import logging


class ConfigDict(dict):
    __getattr__ = dict.__getitem__


def _make_config_dict(obj):
    if isinstance(obj, dict):
        return ConfigDict({k: _make_config_dict(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_make_config_dict(x) for x in obj]
    else:
        return obj


_config = None
_argv_cwd = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def _resolve_repo_path(path):
    if path is None:
        return path
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def _resolve_known_paths(cfg):
    for key in [
        "output_dir",
        "checkpoint",
        "pretrain_astar",
        "pretrain_gen_model",
        "generator_resume_state",
        "resume_state",
    ]:
        if key in cfg:
            cfg[key] = _resolve_repo_path(cfg[key])
    if "dataset" in cfg and "path" in cfg["dataset"]:
        cfg["dataset"]["path"] = _resolve_repo_path(cfg["dataset"]["path"])
    return cfg


def _resolve_config_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_argv_cwd, path))


def config():
    global _config
    if _config is None:
        config_path = 'config.yaml'
        args = sys.argv[1:]
        for i, arg in enumerate(args):
            if arg.startswith('--config='):
                config_path = _resolve_config_path(arg[9:])
                break
            if arg == '--config' and i + 1 < len(args):
                config_path = _resolve_config_path(args[i + 1])
                break
            if arg == '-c' and i + 1 < len(args):
                config_path = _resolve_config_path(args[i + 1])
                break
        print('Reading config from ' + config_path)
        with open(config_path) as f:
            # _config = _make_config_dict(yaml.load(f))
            
            loaded_config = _resolve_known_paths(yaml.load(f, Loader=yaml.FullLoader))
            _config = _make_config_dict(loaded_config)
        overwrite_config_with_args()
    return _config


def path_set(path, val, sep='.', auto_convert=False):
    steps = path.split(sep)
    obj = _config
    for step in steps[:-1]:
        obj = obj[step]
    # print(steps[-1])
    old_val = obj[steps[-1]]
    if not auto_convert:
        obj[steps[-1]] = val
    elif isinstance(old_val, bool):
        obj[steps[-1]] = val.lower() == 'true'
    elif isinstance(old_val, float):
        obj[steps[-1]] = float(val)
    elif isinstance(old_val, int):
        try:
            obj[steps[-1]] = int(val)
        except ValueError:
            obj[steps[-1]] = float(val)
    else:
        obj[steps[-1]] = val


def overwrite_config_with_args(args=None, sep='.'):
    if args is None:
        args = sys.argv[1:]
    for arg in args:
        if arg.startswith('--') and '=' in arg:
            path, val = arg[2:].split('=', 1)
            if path != 'config':
                path_set(path, val, sep, auto_convert=True)


def _dump_config(obj, prefix):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _dump_config(v, prefix + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _dump_config(v, prefix + (str(i),))
    else:
        if isinstance(obj, str):
            rep = obj
        else:
            rep = repr(obj)
        logging.debug('%s=%s', '.'.join(prefix), rep)


def dump_config():
    return _dump_config(_config, tuple())
