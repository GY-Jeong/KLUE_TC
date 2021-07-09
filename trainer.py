import wandb
from sklearn.metrics import accuracy_score

from utils import *


def run(args, model, train_data, valid_data, cv_count):
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
        train_acc, train_loss = train(args, train_loader, model, optimizer)

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


def train(args, train_loader, model, optimizer):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        print(batch)
        exit()
        input = process_batch(batch, args)

        '''
        input 순서는 category + continuous + mask

        'answerCode', 'interaction', 'assessmentItemID', 'testId', 'KnowledgeTag', + 추가 category
        + 추가 cont
        + 'mask'
        '''

        preds = model(input)
        targets = input[0]  # correct
        loss = compute_loss(preds, targets, args)
        update_params(loss, model, optimizer, args)

        if step % args.log_steps == 0:
            print(f"Training steps: {step} Loss: {str(loss.item())}")

        # predictions
        preds = preds[:, -1]
        targets = targets[:, -1]

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
            loss = loss.to('cpu').detach().numpy()
        else:  # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()
            loss = loss.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)
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


