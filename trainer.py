from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn import metrics

import wandb
from utils import *


def run(args, tokenizer, train_data, valid_data, cv_count):
    train_loader, valid_loader = get_loaders(args, train_data, valid_data)

    # only when using warmup scheduler
    args.total_steps = int(len(train_loader.dataset) / args.batch_size) * args.n_epochs
    args.warmup_steps = int(args.total_steps * args.warmup_ratio)

    model = get_model(args)
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
        acc, val_loss = validate(args, model, tokenizer, valid_loader)

        # TODO: model save or early stopping
        if args.scheduler == 'plateau':
            last_lr = optimizer.param_groups[0]['lr']
        else:
            last_lr = scheduler.get_last_lr()[0]

        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
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
    for step, batch in tqdm(enumerate(train_loader), desc='Training', total=len(train_loader)):
        idx, text, label = batch
        label = label.to(args.device)
        # text = ["{} [SEP] {}".format(a_, b_) for a_, b_ in zip(text, summary)]

        tokenized_examples = tokenizer(
            text,
            max_length=args.max_seq_len,
            padding="max_length",
            return_tensors="pt"
        ).to(args.device)

        # tokenize
        # 모델의 입력으로
        # label은 one-hot?
        # loss 주고
        # argmax를 golden

        logits = model(**tokenized_examples)['logits']

        if args.classifier == "CNN" and len(list(logits.shape)) == 1:
            logits = torch.unsqueeze(logits, 0)
        # print(preds)
        # logits = preds['logits']
        # logits = logits[:,0,:]
        # softmax_logits = nn.Softmax(dim=1)(logits)
        argmax_logits = torch.argmax(logits, dim=1)

        # one_hot_logits = one_hot(argmax_logits, num_classes=7).float()
        # print(one_hot(argmax_logits, num_classes=7).type(torch.FloatTensor))
        loss = compute_loss(logits,
                            label, args)

        # print(loss)

        update_params(loss, model, optimizer, step, len(train_loader), args)

        if step % args.log_steps == 0:
            print(f"Training steps: {step} Loss: {str(loss.item())}")
            wandb.log({"steps": step, "train_loss": loss.item()})

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
    print(f'TRAIN ACC : {acc}, TRAIN LOSS : {loss_avg}')
    return acc, loss_avg


def validate(args, model, tokenizer, valid_loader):
    model.eval()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in tqdm(enumerate(valid_loader), desc='Validation', total=len(valid_loader)):
        idx, text, label = batch
        # text = ["{} [SEP] {}".format(a_, b_) for a_, b_ in zip(text, summary)]

        label = label.to(args.device)
        tokenized_examples = tokenizer(
            text,
            max_length=args.max_seq_len,
            padding="max_length",
            return_tensors="pt"
        ).to(args.device)

        # tokenize
        # 모델의 입력으로
        # label은 one-hot?
        # loss 주고
        # argmax를 golden

        logits = model(**tokenized_examples)['logits']

        if args.classifier == "CNN" and len(list(logits.shape)) == 1:
            logits = torch.unsqueeze(logits, 0)

        # softmax_logits = nn.Softmax(dim=1)(logits)
        argmax_logits = torch.argmax(logits, dim=1)

        # one_hot_logits = one_hot(argmax_logits, num_classes=7).float()
        # print(one_hot(argmax_logits, num_classes=7).type(torch.FloatTensor))
        loss = compute_loss(logits,
                            label, args)

        if step % args.log_steps == 0:
            print(f"Validation steps: {step} Loss: {str(loss.item())}")

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
    target_names = ['IT과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치']
    print(metrics.classification_report(total_targets, total_preds, target_names=target_names))
    matrix = metrics.confusion_matrix(total_targets, total_preds)
    print(matrix.diagonal() / matrix.sum(axis=1))

    # Train AUC / ACC
    acc = accuracy_score(total_targets, total_preds)
    loss_avg = sum(losses) / len(losses)
    print(f'VALID ACC : {acc}, VALID LOSS : {loss_avg}')
    return acc, loss_avg


def inference(args, test_data):
    # ckpt_file_names = []
    all_fold_preds = []
    all_fold_argmax_preds = []

    if not args.cv_strategy:
        ckpt_file_names = [args.model_name]
    else:
        ckpt_file_names = [f"{args.model_name.split('.pt')[0]}_{i + 1}.pt" for i in range(args.fold_num)]

    tokenizer = load_tokenizer(args)

    for fold_idx, ckpt in enumerate(ckpt_file_names):
        model = load_model(args, ckpt)
        model.eval()
        test_loader = get_loaders(args, None, test_data, True)

        total_preds = []
        total_argmax_preds = []
        total_ids = []

        for step, batch in tqdm(enumerate(test_loader), desc='Inferencing', total=len(test_loader)):
            idx, text = batch

            # text = ["{} [SEP] {}".format(a_, b_) for a_, b_ in zip(text, summary)]

            tokenized_examples = tokenizer(
                text,
                max_length=args.max_seq_len,
                padding="max_length",
                return_tensors="pt"
            ).to(args.device)

            # preds = model(**tokenized_examples)

            logits = model(**tokenized_examples)['logits']

            if args.classifier == "CNN" and len(list(logits.shape)) == 1:
                logits = torch.unsqueeze(logits, 0)

            argmax_logits = torch.argmax(logits, dim=1)

            if args.device == 'cuda':
                argmax_preds = argmax_logits.to('cpu').detach().numpy()
                preds = logits.to('cpu').detach().numpy()
            else:  # cpu
                argmax_preds = argmax_logits.detach().numpy()
                preds = logits.detach().numpy()

            total_preds += list(preds)
            total_argmax_preds += list(argmax_preds)
            total_ids += list(idx)

        all_fold_preds.append(total_preds)
        all_fold_argmax_preds.append(total_argmax_preds)

        output_file_name = "output.csv" if not args.cv_strategy else f"output_{fold_idx + 1}.csv"
        write_path = os.path.join(args.output_dir, output_file_name)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(write_path, 'w', encoding='utf8') as w:
            print("writing prediction : {}".format(write_path))
            w.write("index,topic_idx\n")
            for index, p in zip(total_ids, total_argmax_preds):
                w.write('{},{}\n'.format(index, p))

    if len(all_fold_preds) > 1:
        # Soft voting ensemble
        votes = np.sum(all_fold_preds, axis=0)
        votes = np.argmax(votes, axis=1)

        write_path = os.path.join(args.output_dir, "output_softvote.csv")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(write_path, 'w', encoding='utf8') as w:
            print("writing prediction : {}".format(write_path))
            w.write("index,topic_idx\n")
            for id, p in zip(total_ids, votes):
                w.write('{},{}\n'.format(id, p))
