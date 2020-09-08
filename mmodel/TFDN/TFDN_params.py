import argparse
from ..basic_params import basic_parser

parser = argparse.ArgumentParser(parents=[basic_parser], conflict_handler="resolve")

parser.add_argument("-cw", action="store_false", dest="cls_wise_accu")

parser.add_argument("--steps", type=int, default=100000)

parser.add_argument("--lr", type=float, default=0.01)

parser.add_argument("--c_ent", type=float, default=0.3)

parser.add_argument("--c_norm", type=float, default=0.01)

parser.add_argument("--dataset", type=str, default="OFFICEHOME")

parser.add_argument("--source", type=str, default="Rw")

parser.add_argument("--target", type=str, default="Cl")

params = parser.parse_args()

