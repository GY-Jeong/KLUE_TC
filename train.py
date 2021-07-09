import torch
from sklearn.model_selection import KFold, StratifiedKFold
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import wandb

import trainer
from args import parse_args
from dataloader import Preprocess
from utils import set_seeds


def main(args):
    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        args.config_name
        if args.config_name
        else args.model_name_or_path,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name
        if args.tokenizer_name
        else args.model_name_or_path,
        use_fast=True,
    )

    config.num_labels = 7
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    preprocess = Preprocess(args)
    preprocess.load_train_data()
    train_data_origin = preprocess.train_data

    print(f"Size of train data : {len(train_data_origin)}")
    # print(f"size of test data : {len(test_data)}")

    if args.cv_strategy == 'random':
        kf = KFold(n_splits=args.fold_num, shuffle=True)
        splits = kf.split(X=train_data_origin)
    else:
        # default
        # 여기 각 label로 바꿔야됨
        train_labels = [sequence[-1] for sequence in train_data_origin]
        skf = StratifiedKFold(n_splits=args.fold_num, shuffle=True)
        splits = skf.split(X=train_data_origin, y=train_labels)

    acc_avg = 0
    for fold_num, (train_index, valid_index) in enumerate(splits):
        train_data = train_data_origin[train_index]
        valid_data = train_data_origin[valid_index]
        best_acc = trainer.run(args, model, train_data, valid_data, fold_num + 1)

        if not args.cv_strategy:
            break

        acc_avg += best_acc

    if args.cv_strategy:
        acc_avg /= args.kfold_num
        wandb.log({"auc_avg": acc_avg})

        print("*" * 50, 'auc_avg', "*" * 50)
        print(acc_avg)


if __name__ == '__main__':
    args = parse_args()
    main(args)
