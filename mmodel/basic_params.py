import argparse
from mtrain.watcher import watcher


_basic_params = None

basic_parser = argparse.ArgumentParser(description="basic", add_help=False)

basic_parser.add_argument("-m", action="store_false", dest="disable_log")

basic_parser.add_argument("-r", action="store_false", dest="make_record")

basic_parser.add_argument("--tag", type=str, default="NO TAG")

basic_parser.add_argument("--steps", type=int)

basic_parser.add_argument("-cw", action="store_true", dest="cls_wise_accu")

basic_parser.add_argument("--use_gpu", type=bool, default=True)

basic_parser.add_argument("--log_per_step", type=int, default=2)

basic_parser.add_argument("--eval_per_step", type=int, default=200)

basic_parser.add_argument("--num_workers", type=int, default=4)

basic_parser.add_argument("--batch_size", type=int, default=64)

basic_parser.add_argument("--eval_batch_size", type=int, default=64)

basic_params = basic_parser.parse_args()
