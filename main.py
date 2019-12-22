
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="TriMine")
    parser.add_argument('-a', '--alpha', type=float,
        help='scale of alpha for Object Matrix')
    parser.add_argument('-b', '--beta', type=float,
        help='scale of beta for Actor Matrix')
    parser.add_argument('-g', '--gamma', type=float,
        help='scale of gamma for Time matrix')
    parser.add_argument('-o', '--outdir', type=str,
        default='_out/tmp/')
    parser.add_argument('--setseed',
        action='store_true')

    args = parser.parse_args()

