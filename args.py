import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', default='temp', type=str, help='wandb run name. Defaults to current time')

    parser.add_argument('--seed', default=42, type=int, help='seed')

    parser.add_argument('--device', default='cuda', type=str, help='cpu or gpu')

    parser.add_argument('--data_dir', default='./data/open/', type=str, help='data directory')

    parser.add_argument('--model_dir', default='models/', type=str, help='model directory')
    parser.add_argument('--model_name_or_path', default='monologg/koelectra-base-v3-discriminator', type=str,
                        help='model file name')
    parser.add_argument('--config_name', default=None, type=str, help='model config name')
    parser.add_argument('--tokenizer_name', default=None, type=str, help='model tokenizer name')

    parser.add_argument('--cv_strategy', default='stratified', type=str, help='choose cross validation strategy')
    parser.add_argument('--fold_num', default=5, type=int, help='')

    parser.add_argument('--num_workers', default=1, type=int)

    # 훈련
    parser.add_argument('--n_epochs', default=10, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--clip_grad', default=10, type=int, help='clip grad')
    parser.add_argument('--patience', default=5, type=int, help='for early stopping')
    parser.add_argument('--max_seq_len', default=40, type=int, help='for early stopping')

    # Optimizer
    parser.add_argument('--optimizer', default='adam', type=str, help='optimizer type')

    # Optimizer-parameters
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay of optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, help="momentum for SGD")

    # Scheduler
    parser.add_argument('--scheduler', default='plateau', type=str, help='scheduler type')

    # Scheduler-parameters
    # plateau
    parser.add_argument('--plateau_patience', default=10, type=int, help='patience of plateau scheduler')
    parser.add_argument('--plateau_factor', default=0.5, type=float, help='factor of plateau scheduler')

    parser.add_argument('--criterion', default='MSE', type=str, help='criterion type')

    parser.add_argument('--log_steps', default=1, type=int, help='print log per n steps')

    args = parser.parse_args()

    return args
