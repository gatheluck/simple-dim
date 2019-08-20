from options import *
from data import *

import torch
import torch.optim as optim

import tqdm

if __name__ == '__main__':
    opt = TrainOptions().parse()

    # dataset
    train_dataset = get_dataset(opt.dataset, train=True, input_size=opt.input_size, num_samples=opt.num_samples)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=opt.cuda)

    # model
    model = Encoder().to(opt.device)
    
    # optimizer
    model_optimizer = optim.Adam(model.parameters(), lr = opt.lr)
    loss_optimizer  = optim.Adam(dim_loss.parameters(), lr = opt.lr)
    labels = get_labels(opt.dataset)
    opt.num_classes = len(labels)

    for epoch in tqdm.tqdm(range(opt.num_epochs)):
    

