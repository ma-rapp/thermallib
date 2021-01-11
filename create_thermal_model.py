import argparse
import os
import sys
import re
from chip import Chip


def amount_cores_type(s):
    pat = re.compile(r"(\d+)x(\d+)")
    m = pat.match(s)
    if not m:
        raise argparse.ArgumentTypeError('invalid format. Valid examples: 3x4, 8x8')
    return (int(m.group(1)), int(m.group(2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="folder where to create the thermal model")
    parser.add_argument("cores", help="amount of cores, e.g.: 8x8", type=amount_cores_type)
    parser.add_argument("--amb", help="ambient temperature", type=float, default=45)
    parser.add_argument("--crit", help="critical temperature", type=float, default=80)
    parser.add_argument("--core_naming", help="core naming", default='sniper', choices=('sniper', 'xy'))
    parser.add_argument("--tdp", help="TDP (provide either TDP or core width)", type=float, default=None)
    parser.add_argument("--core_width", help="core width in meters (provide either TDP or core width)", type=float, default=None)
    parser.add_argument("--create_matlab_code", help="This is self-explaining :)", dest='create_matlab_code', action='store_true')
    args = parser.parse_args()

    folder = args.folder
    processorsY, processorsX = args.cores
    ambient_temperature = args.amb
    max_temperature = args.crit
    inactive_power = 0
    core_naming = args.core_naming
    config = None
    tdp = args.tdp
    core_width = args.core_width
    if (tdp and core_width) or not (tdp or core_width):
        print('provide either TDP or core width')
        sys.exit(1)

    if tdp:
        chip = Chip.load_or_create(folder, processorsY, processorsX, tdp, ambient_temperature, max_temperature, inactive_power, core_naming, config)
    elif core_width:
        chip = Chip.load_or_create_fixed_size(folder, processorsY, processorsX, core_width, ambient_temperature, max_temperature, inactive_power, core_naming, config)
        print('TDP: {:.0f}W'.format(chip.tdp()))
    if args.create_matlab_code:
        chip.create_matlab_code()

if __name__ == "__main__":
    main()
