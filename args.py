import argparse


def parse_args(mode):
    parser = argparse.ArgumentParser()
    if mode == 'train':
        parser.add_argument('--run_name', default=None, type=str, help='wandb run name. Defaults to current time')
    else:
        parser.add_argument('--model_name', default=None, type=str)
        parser.add_argument('--output_dir', default='output/', type=str)

    parser.add_argument('--seed', default=42, type=int, help='seed')

    parser.add_argument('--device', default='cuda', type=str, help='cpu or gpu')

    parser.add_argument('--data_dir', default='./data/open/', type=str, help='data directory')

    parser.add_argument('--classifier', default='CNN', type=str, help='model directory')
    parser.add_argument('--model_dir', default='models/', type=str, help='model directory')
    parser.add_argument('--model_name_or_path', default='klue/roberta-large', type=str,
                        help='model file name')
    parser.add_argument('--config_name', default=None, type=str, help='model config name')
    parser.add_argument('--tokenizer_name', default=None, type=str, help='model tokenizer name')

    # CNN config
    parser.add_argument('--out_size', default=32, type=int, help='')
    parser.add_argument('--stride', default=2, type=int, help='')

    parser.add_argument('--accum_iter', default=16, type=int, help='')
    parser.add_argument('--gradient_accumulation', default=True, type=str, help='')

    parser.add_argument('--cv_strategy', default='stratified', type=str, help='choose cross validation strategy')
    parser.add_argument('--fold_num', default=4, type=int, help='')

    parser.add_argument('--num_workers', default=1, type=int)

    # 훈련
    parser.add_argument('--n_epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--lr', default=5e-6, type=float, help='learning rate')
    parser.add_argument('--clip_grad', default=15, type=int, help='clip grad')
    parser.add_argument('--patience', default=5, type=int, help='for early stopping')
    parser.add_argument('--max_seq_len', default=40, type=int, help='for early stopping')

    # Optimizer
    parser.add_argument('--optimizer', default='adamP', type=str, help='optimizer type')

    # Optimizer-parameters
    parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay of optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, help="momentum for SGD")

    # Scheduler
    parser.add_argument('--scheduler', default='step_lr', type=str, help='scheduler type')

    # Scheduler-parameters
    # plateau
    parser.add_argument('--plateau_patience', default=10, type=int, help='patience of plateau scheduler')
    parser.add_argument('--plateau_factor', default=0.5, type=float, help='factor of plateau scheduler')

    # cosine anealing
    parser.add_argument('--t_max', default=100, type=int, help='cosine annealing scheduler: t max')
    parser.add_argument('--T_0', default=10, type=int, help='cosine annealing warm start scheduler: T_0')
    parser.add_argument('--T_mult', default=2, type=int, help='cosine annealing warm start scheduler: T_mult')
    parser.add_argument('--eta_min', default=0.0, type=float, help='cosine annealing warm start scheduler: eta_min')

    # linear_warmup
    parser.add_argument('--warmup_ratio', default=0.3, type=float, help='warmup step ratio')

    # Step LR
    parser.add_argument('--step_size', default=50, type=int, help='step LR scheduler: step size')
    parser.add_argument('--gamma', default=0.1, type=float, help='step LR scheduler: gamma')

    parser.add_argument('--criterion', default='CE', type=str, help='criterion type')

    parser.add_argument('--log_steps', default=100, type=int, help='print log per n steps')

    args = parser.parse_args()

    return args
