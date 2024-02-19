import argparse

from typing import Union


def parse_args_by_group(parser: argparse.ArgumentParser, group_positional: bool = True, group_optional: bool = True) -> argparse.Namespace:
    """Similar to `ArgumentParser.parse_args()` but returns a nested `Namespace` that has the groups of the 
    `ArgumentParser` at the top-level.

    Can be useful to monkey-patch an ArgumentParser object like:
        `parser.parse_args_by_group = functools.partial(parse_args_by_group, parser=parser)`
    """
    args = parser.parse_args()

    groups = dict()
    for group in parser._action_groups:
        group_kwargs = {action.dest: getattr(args, action.dest, None) for action in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_kwargs)

    del_keys = []
    if not group_positional:
        del_keys.append("positional arguments")
    if not group_optional:
        del_keys.append("optional arguments")
    if del_keys:
        for dk in del_keys:
            kwargs = vars(groups[dk])
            del groups[dk]
            for k, v in kwargs.items():
                groups[k] = v

    return argparse.Namespace(**groups)


def int_or_str(arg: Union[int, str]) -> Union[int, str]:
    """Parse a string argument to int if it looks like int else return the string."""
    try:
        return int(arg)
    except ValueError:
        return arg


def float_or_str(arg: Union[float, str]) -> Union[float, str]:
    """Parse a string argument to float if it looks like float else return the string."""
    try:
        return float(arg)
    except ValueError:
        return arg


def str2bool(arg: Union[bool, str]) -> bool:
    """Parse a string argument to bool.

    To be used as:
        parser.add_argument('--some_var', type=str2bool, default=False)

    Arguments parsed to 'True' (case insensitive):
        --some_var true
        --some_var t
        --some_var yes
        --some_var y
        --some_var 1

    Arguments parsed to 'False' (case insensitive):
        --some_var false
        --some_var f
        --some_var no
        --some_var n
        --some_var 0

    See https://stackoverflow.com/a/43357954/4203328
    """
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif arg.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Could not parse argument {arg} of type {type(arg)} as bool.")
