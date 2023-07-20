import argparse
from yacs.config import CfgNode as CN

from engine import Tester


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode', type=str, choices=['evaluate', 'sample'],
        help='Choose to evaluate or sample',
    )
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='Path to configuration file',
    )
    parser.add_argument(
        '-ni', '--no-interaction', action='store_true', default=False,
        help='Do not interact with user (always choose yes when interacting)',
    )
    parser.add_argument(
        '--downstream', action='store_true', default=False,
        help='Test downtream tasks',
    )
    return parser


if __name__ == '__main__':
    args, unknown_args = get_parser().parse_known_args()
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.config)
    cfg.set_new_allowed(False)
    cfg.merge_from_list(unknown_args)
    cfg.freeze()

    if args.downstream:
        from engine.tester_downstream import DownstreamTester
        tester = DownstreamTester(args, cfg)
    else:
        tester = Tester(args, cfg)

    if args.mode == 'evaluate':
        tester.evaluate()
    else:
        tester.sample()
