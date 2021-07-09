import wandb
from sklearn.metrics import accuracy_score
from torch.nn.functional import one_hot

from utils import *


def run(args, model, tokenizer, train_data, valid_data, cv_count):
    train_loader, valid_loader = get_loaders(args, train_data, valid_data)

    # only when using warmup scheduler
    # args.total_steps = int(len(train_loader.dataset) / args.batch_size) * args.n_epochs
    # args.warmup_steps = int(args.total_steps * args.warmup_ratio)

    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    best_acc = -1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):

        print(f"Start Training: Epoch {epoch + 1}")

        if not args.cv_strategy:
            model_name = args.run_name
        else:
            model_name = f"{args.run_name.split('.pt')[0]}_{cv_count}.pt"

        # TRAIN
        train_acc, train_loss = train(args, model, tokenizer, train_loader, optimizer)

        # VALID
        acc, _, _, val_loss = validate(args, valid_loader, model)

        # TODO: model save or early stopping
        if args.scheduler == 'plateau':
            last_lr = optimizer.param_groups[0]['lr']
        else:
            last_lr = scheduler.get_last_lr()[0]

        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                   "valid_acc": acc, "val_loss": val_loss, "learning_rate": last_lr})

        if acc > best_acc:
            best_acc = acc
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, 'module') else model
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
            },
                args.model_dir, model_name,
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(f'EarlyStopping counter: {early_stopping_counter} out of {args.patience}')
                break

        # scheduler
        if args.scheduler == 'plateau':
            scheduler.step(best_acc)
        else:
            scheduler.step()

    return best_acc


def train(args, model, tokenizer, train_loader, optimizer):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        idx, text, label = batch

        tokenized_examples = tokenizer(
            text,
            max_length=args.max_seq_len,
            padding="max_length",
            return_tensors="pt"
        )

        # tokenize
        # 모델의 입력으로
        # label은 one-hot?
        # loss 주고
        # argmax를 golden

        preds = model(**tokenized_examples)
        logits = preds['logits']
        argmax_logits = torch.argmax(logits, dim=1)
        loss = compute_loss(logits, one_hot(label).type(torch.FloatTensor), args)
        update_params(loss, model, optimizer, args)

        if step % args.log_steps == 0:
            print(f"Training steps: {step} Loss: {str(loss.item())}")

        if args.device == 'cuda':
            argmax_logits = argmax_logits.to('cpu').detach().numpy()
            label = label.to('cpu').detach().numpy()
            loss = loss.to('cpu').detach().numpy()
        else:  # cpu
            argmax_logits = argmax_logits.detach().numpy()
            label = label.detach().numpy()
            loss = loss.detach().numpy()

        total_preds.append(argmax_logits)
        total_targets.append(label)
        losses.append(loss)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    acc = accuracy_score(total_targets, total_preds)
    loss_avg = sum(losses) / len(losses)
    print(f'TRAIN ACC : {acc}')
    return acc, loss_avg


def validate(args, valid_loader, model):
    pass


