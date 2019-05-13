import torch
import torch.nn as nn
from APN import APN
from torch.utils.data import TensorDataset, DataLoader


def get_batch_train_val_acc(pos_score, neg_score):
    correct = torch.sum(pos_score > neg_score).detach().item()
    acc = correct / torch.numel(pos_score)
    return acc


def run(model, loss_fn, optimizer, device, max_epoch, log_freq, train_set, val_set, test_set, num_workers,
        train_batch_size,
        test_val_batch_size, threshold, metrics):
    # train mode
    if train_set and val_set and not test_set:
        train_loader = DataLoader(train_set, train_batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_set, test_val_batch_size, shuffle=False, num_workers=num_workers)
        model.to(device)
        global_step = 0
        best_acc = 0
        best_epoch = 0
        best_model_state = model.state_dict()
        for e in range(max_epoch):
            print('*' * 10 + 'Epoch ' + str(e) + '/' + str(max_epoch) + '*' * 10)

            for input in train_loader:
                global_step += 1
                for i in range(len(input)):
                    input[i] = input[i].to(device)
                label = input[-1]
                pos_score, neg_score = model(*input[:-1])
                optimizer.zero_grad()
                loss = loss_fn(pos_score, neg_score, label)
                loss.backward()
                optimizer.step()

                # log stats and save model ckpt
                if global_step % log_freq == 0:
                    model.eval()
                    if 'acc' in metrics:
                        train_acc = get_batch_train_val_acc(pos_score, neg_score)
                        val_loss, val_acc = [], []
                    for input in val_loader:
                        for i in range(len(input)):
                            input[i] = input[i].to(device)
                        label = input[-1]
                        pos_score, neg_score = model(*input[:-1])
                        batch_val_loss = loss_fn(pos_score, neg_score, label)
                        if 'acc' in metrics:
                            batch_val_acc = get_batch_train_val_acc(pos_score, neg_score)
                            val_loss.append(batch_val_loss.detach().item())
                            val_acc.append(batch_val_acc)
                    val_loss = sum(val_loss) / len(val_loss)
                    if 'acc' in metrics:
                        val_acc = sum(val_acc) / len(val_acc)
                        print(
                            f'E {e}\tB {global_step}\tt_loss {loss:.4f}\tv_loss {val_loss:.4f}\tt_acc {train_acc:.4f}\tv_acc {val_acc:.4f}')
                    if 'acc' in metrics and val_acc > best_acc:
                        best_model_state = model.state_dict()
                        best_epoch = e
                    model.train()
        torch.save({'best_epoch': best_epoch, 'best_model_state': best_model_state,
                    'optim_state': optimizer.state_dict()}, f'data/val_acc{best_acc:.4f}.ckpt')
    elif test_set:
        # test mode
        ckpt = torch.load('data/val_acc0.0000.ckpt')
        model.load_state_dict(ckpt['best_model_state'])

        test_loader = DataLoader(test_set, test_val_batch_size, shuffle=False, num_workers=num_workers)
        label = []
        for q, a in test_loader:
            simi_score = model(q, a)
            label += [score.item() for score in simi_score]
            label = [int(score > threshold) * 2 - 1 for score in label]
            with open('data/pred_test.tsv', 'w') as f:
                for la in label:
                    f.write(str(la) + '\n')
    return


if __name__ == '__main__':
    embed_size = 20
    embed_dim = 100
    embed_path = False
    encoder_type = 'rnn'
    rnn_hidden_size = 30
    rnn_bidirectional = True
    rnn_num_layer = 1
    cnn_num_layer = None
    cnn_kernel_sizes = None
    cnn_num_kernel = None
    model = APN(embed_size, embed_dim, embed_path,
                encoder_type,
                rnn_hidden_size, rnn_bidirectional, rnn_num_layer,
                cnn_num_layer, cnn_kernel_sizes, cnn_num_kernel
                )

    max_epoch = 10
    log_freq = 5

    len_train_set = 70
    len_val_set = 20
    q_len = 22
    a_len = 33
    neg_sample_rate = 10
    train_set = TensorDataset(torch.randint(embed_size, (len_train_set, q_len), dtype=torch.long),
                              torch.randint(embed_size, (len_train_set, a_len), dtype=torch.long),
                              *([torch.randint(embed_size, (len_train_set, a_len),
                                               dtype=torch.long)] * neg_sample_rate), torch.ones(len_train_set))
    val_set = TensorDataset(torch.randint(embed_size, (len_val_set, q_len), dtype=torch.long),
                            torch.randint(embed_size, (len_val_set, a_len), dtype=torch.long),
                            *([torch.randint(embed_size, (len_val_set, a_len), dtype=torch.long)] * neg_sample_rate),
                            torch.ones(len_val_set))
    test_set = TensorDataset(torch.randint(embed_size, (len_val_set, q_len), dtype=torch.long),
                             torch.randint(embed_size, (len_val_set, a_len), dtype=torch.long))

    train_batch_size = 2
    test_val_batch_size = 128

    metrics = ['acc']

    loss_fn = nn.MarginRankingLoss(margin=1)
    optimizer = torch.optim.Adam(model.parameters())
    device = 'cuda'
    threshold = 0.5
    num_workers = 4

    run(model, loss_fn, optimizer, device, max_epoch, log_freq, None, None, test_set, num_workers,
        train_batch_size,
        test_val_batch_size, threshold, metrics)
    # first train log trian stats save best model then test save predict res
