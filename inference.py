import torch

import trainer
from args import parse_args
from dataloader import Preprocess
from utils import set_seeds


def main(args):
    set_seeds(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
    preprocess.load_test_data()
    test_data = preprocess.test_data

    print(f"size of test data : {len(test_data)}")

    trainer.inference(args, test_data)


if __name__ == '__main__':
    args = parse_args('inference')
    main(args)
