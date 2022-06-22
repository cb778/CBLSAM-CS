import torch
import torch.optim as optim
import os
import time
import numpy as np
import argparse
import configs
from loguru import logger
from dataLoader import DataLoader
import attentionNet as jointemb
from utils import validate

def train_epoch(model, train_data, validation_data, optimizer, config, args,  device):
    model.train()
    losses = []
    itr_start_time = time.time()
    n_itr = len(train_data)
    text = "---------- Training ----------"
    def save_model(model, path):
        torch.save(model.state_dict(), path)

    for index, batch in enumerate(train_data):
        model.train()
        batch = [data.to(device) for data in batch][0::2]


        optimizer.zero_grad()
        loss = model(*batch)

        if config['fp16']:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 5.0)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()
        losses.append(loss.item())

        if (index+1) % args.log_every == 0:
            elapsed = time.time() - itr_start_time
            info = 'itr:{} step_time:{} Loss={}'.format((index+1), elapsed/(index+1), np.mean(losses))
            logger.info(info)

        if (index+1) % args.valid_every == 0:
            logger.info('Validating.')
            re, acc, mrr, map, ndcg = validate(validation_data, model, 1, 'cos')
            result = 'Recall={}, Accurate={}, Mrr={}, Map={}, NDCG={}'.format(re, acc, mrr, map, ndcg)
            print(result)

    return losses


def train(model, training_data, validation_data, optimizer, args, device):
    print("The model will train epoch is: ", args.Epoch)
    logger.info("Start training!")
    best_mrr = 0.
    config = getattr(configs, 'config')()

    def save_model(model, path):
        torch.save(model.state_dict(), path)

    def adjust_learning_rate(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    for epoch in range(args.Epoch):
        info = '[ Epoch ' + str(epoch) + ' ]'
        logger.info(info)
        lr = config['learning_rate'] * (0.6 ** (epoch // 4))
        adjust_learning_rate(optimizer, lr)
        print('当前epoch的学习率是：', lr)

        train_loss = train_epoch(model, training_data, validation_data, optimizer, config, args, device)
        logger.info("The loss of epoch {} is: {}".format(epoch, np.mean(train_loss)))
        logger.info("Validating.")
        re, acc, mrr, _, _ = validate(validation_data, model, 1, config['sim_measure'])
        valid_mrr = mrr
        if best_mrr < valid_mrr:
            best_mrr = valid_mrr
            print("The current best mrr score is: ", best_mrr)
            path = args.model_path + 'lstm_attn_v1_bigData(bathsize=128).h5'
            save_model(model, path)
        else:
            print("Skip this step...and the current mrr value is: ", valid_mrr)
            print("The model have the best mrr value is: ", best_mrr)

        epoc_path = f'./output/step_{epoch}.h5'
        torch.save(model.state_dict(), epoc_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../dataset/', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='JointEmbedder', help='model name')
    parser.add_argument('--dataset', type=str, default='smallData/', help='name of dataset.java')
    parser.add_argument('--reload_from', type=int, default=-1, help='epoch to reload from')
    parser.add_argument('--model_path', type=str, default='./model_save/', help='path of saving model')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('-v', "--visual", action="store_true", default=False, help="Visualize training status in tensorboard")
    parser.add_argument('--best_mrr', type=float, default=0., help='The MRR metric.')
    parser.add_argument('--log_every', type=int, default=1000, help='interval to log autoencoder training results')
    parser.add_argument('--valid_every', type=int, default=30000, help='interval to validation')
    parser.add_argument('--save_every', type=int, default=10000, help='interval to evaluation to concrete results')
    parser.add_argument('--sim_measure', type=str, default='cos', help='similarity measure for training')
    parser.add_argument('--Epoch', type=int, default=15, help="Training Epoch")

    args = parser.parse_args()
    config = getattr(configs, 'config')()
    print(config)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # ------ Load Dataset ---------
    data_path = args.data_path + args.dataset

    valid_set = DataLoader(data_path, config['valid_name'], config['name_len'],
                        config['valid_api'], config['api_len'], config['valid_tokens'],
                        config['tokens_len'], config['valid_desc'], config['desc_len'])

    model = getattr(jointemb, args.model)(config)  # JointEmbedder
    optimizer = optim.AdamW(model.parameters() ,lr=config['learning_rate'], eps=config['adam_epsilon'])



    def load_model(model, ckpt_path, to_device):
        assert os.path.exists(ckpt_path), f'Weights not found'
        model.load_state_dict(torch.load(ckpt_path, map_location=to_device))

    def count_parameters(model):
       return sum((p.numel() for p in model.parameters() if p.requires_grad))

    print(f'The model has {count_parameters(model):,} trainable parameters!')

    if config["fp16"]:
        try:
            from apex import amp
        except:
            ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=config['fp16_opt_level'])

    if args.mode == 'train':
        train_set = DataLoader(data_path, config['train_name'], config['name_len'],
                            config['train_api'], config['api_len'], config['train_tokens'],
                            config['tokens_len'], config['train_desc'], config['desc_len'])
        train_iter = torch.utils.data.DataLoader(dataset=train_set, batch_size=config['batch_size'],
                                                 shuffle=True, drop_last=True, num_workers=1)
        model = model.to(device)
        with open('lstm_attn_v1_mode.txt', 'w') as f:
            f.write(str(model))

        train(model, train_iter, valid_set, optimizer, args, device)


    elif args.mode == 'eval':
        path = args.model_path + 'xxxx.h5'
        load_model(model, path, device)
        model.to(device)
        start_time = time.time()
        K = [1, 5, 10]
        for k in K:
            re, acc, mrr, map, ndcg = validate(valid_set, model, k, 'cos')
            search_time = time.time() - start_time
            query_time = search_time / 10000
            results = "k={}, re={}, acc={}, mrr={}, ndcg={}, q_time={}".format(k, re, acc, mrr, ndcg, query_time)
            with open('result.txt', 'a') as f:
                f.write(results + '\n')
            print("The search time of each query is: {}".format(query_time))



if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # speed up training by using cudnn
    main()





