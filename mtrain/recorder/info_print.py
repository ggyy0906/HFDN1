"""
ref to https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-terminal-in-python
"""
HEADER_SIGN = "⯈"

FMT = "\x1b[6;30;4{c}m \x1b[0m\x1b[6;3{c};40m ⯈\x1b[0m \x1b[1;3{c};40m {{s}}\x1b[0m"

WARMN = FMT.format(c=1)
TRAIN = FMT.format(c=2)
VALID = FMT.format(c=3)
HINTS = FMT.format(c=4)
BUILD = FMT.format(c=5)


def cprint(f, context):
    print(f.format(s=context))


from tabulate import tabulate
from collections import defaultdict

losses_history = {"train": list(), "valid": list()}


def get_changing_str(number):
    arrow = "⮥" if number > 0 else "⮧"
    return arrow + "  " + "%.5f" % (abs(number))


def tabulate_print_losses(losses, mode="train"):
    assert mode in ["train", "valid"]

    historys = losses_history[mode]

    if len(historys) == 0:
        items = [(k, losses[k], "NONE") for k in losses]
    else:
        last_losses = historys[-1]
        items = [
            (k, losses[k], get_changing_str(losses[k] - last_losses[k]))
            for k in losses
        ]

    historys.append(losses)
    table = tabulate(items, tablefmt="grid").split("\n")
    log_mode = TRAIN if mode is "train" else VALID
    for line in table:
        cprint(log_mode, line)


if __name__ == "__main__":
    cprint(TRAIN, "aaa ")
    cprint(TRAIN, "dd")
    cprint(TRAIN, "ff")
    cprint(VALID, "aaaa")
    cprint(WARMN, "aaaa")
    cprint(HINTS, "aaaa")

