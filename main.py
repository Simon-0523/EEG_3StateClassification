from cross_validation import *
from prepare_data import *
import argparse
import sys
from terminal_output import Logger

sys.stdout = Logger("output.txt")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ######## Data ########
    parser.add_argument('--dataset', type=str, default='state')
    parser.add_argument('--data-path', type=str, default='D:\\project\\github\\test_muse')
    parser.add_argument('--subjects', type=int, default=1)
    parser.add_argument('--num-class', type=int, default=3, choices=[2, 3, 4])
    parser.add_argument('--segment', type=int, default=4, help='segment length in seconds')
    parser.add_argument('--trial-duration', type=int, default=60, help='trial duration in seconds')
    parser.add_argument('--num_segment', type=int, default=0)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--sampling-rate', type=int, default=256)
    parser.add_argument('--input-shape', type=tuple, default=(1, 4, 256*4))
    parser.add_argument('--data-format', type=str, default='raw')
    ######## Training Process ########
    parser.add_argument('--random-seed', type=int, default=12)
    parser.add_argument('--max-epoch', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=128) 
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--save-path', default='./save/')
    parser.add_argument('--load-path', default='./save/max-acc.pth')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--save-model', type=bool, default=True)
    ######## Model Parameters ########
    parser.add_argument('--model', type=str, default='State_Detection')
    parser.add_argument('--T', type=int, default=16)
    parser.add_argument('--hidden', type=int, default=4)

    ######## Reproduce the result using the saved model ######
    parser.add_argument('--reproduce', action='store_true')
    args = parser.parse_args()

    sub_to_run = np.arange(args.subjects)

    # pd = PrepareData(args)
    # pd.run(sub_to_run, split=True, feature=False, expand=True)

    cv = CrossValidation(args)
    seed_all(args.random_seed)
    cv.n_fold_CV(subject=sub_to_run, fold=5, reproduce=args.reproduce)  # To do leave one trial out please set fold=40
# sys.stdout.log.close()