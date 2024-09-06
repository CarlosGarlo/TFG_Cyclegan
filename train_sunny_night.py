"""
Training for CycleGAN

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-05: Initial coding
* 2022-12-21: Small revision of code, checked that it works with latest PyTorch version
"""

import torch

import dataset
from dataset import SunnyNightDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
import numpy as np
import matplotlib.pyplot as plt

def train_fn(
    disc_S, disc_N, gen_N, gen_S, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    S_reals = 0
    S_fakes = 0
    loop = tqdm(loader, leave=True)
    # Listas vacias para plotear graficas
    cycle_night_loss_list = []
    cycle_sunny_loss_list = []
    D_loss_list = []
    # G_loss_list = []

    for idx, (night, sunny) in enumerate(loop):
        night = night.to(config.DEVICE)
        sunny = sunny.to(config.DEVICE)

        # Train Discriminators S and N
        with torch.cuda.amp.autocast():
            fake_sunny = gen_S(night)
            D_S_real = disc_S(sunny)
            D_S_fake = disc_S(fake_sunny.detach())
            S_reals += D_S_real.mean().item()
            S_fakes += D_S_fake.mean().item()
            D_S_real_loss = mse(D_S_real, torch.ones_like(D_S_real))
            D_S_fake_loss = mse(D_S_fake, torch.zeros_like(D_S_fake))
            D_S_loss = D_S_real_loss + D_S_fake_loss

            fake_night = gen_N(sunny)
            D_N_real = disc_N(night)
            D_N_fake = disc_N(fake_night.detach())
            D_N_real_loss = mse(D_N_real, torch.ones_like(D_N_real))
            D_N_fake_loss = mse(D_N_fake, torch.zeros_like(D_N_fake))
            D_N_loss = D_N_real_loss + D_N_fake_loss

            # put it togethor
            D_loss = (D_S_loss + D_N_loss) / 2
            D_loss_array = D_loss.cpu().detach().numpy()

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators S and N
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_S_fake = disc_S(fake_sunny)
            D_N_fake = disc_N(fake_night)
            loss_G_S = mse(D_S_fake, torch.ones_like(D_S_fake))
            loss_G_N = mse(D_N_fake, torch.ones_like(D_N_fake))

            # cycle loss
            cycle_night = gen_N(fake_sunny)
            cycle_sunny = gen_S(fake_night)
            cycle_night_loss = l1(night, cycle_night)
            cycle_night_loss_array = cycle_night_loss.cpu().detach().numpy()
            cycle_sunny_loss = l1(sunny, cycle_sunny)
            cycle_sunny_loss_array = cycle_sunny_loss.cpu().detach().numpy()
            # identity loss (remove these for efficiency if you set lambda_identity=0)
            # identity_night = gen_N(night)
            # identity_sunny = gen_S(sunny)
            # identity_night_loss = l1(night, identity_night)
            # identity_sunny_loss = l1(sunny, identity_sunny)

            # add all togethor
            G_loss = (
                loss_G_N
                + loss_G_S
                + cycle_night_loss * config.LAMBDA_CYCLE
                + cycle_sunny_loss * config.LAMBDA_CYCLE
                # + identity_sunny_loss * config.LAMBDA_IDENTITY
                # + identity_night_loss * config.LAMBDA_IDENTITY
            )
           # G_loss_array = G_loss.detach().to('cpu').numpy()

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_sunny, f"saved_images/sunny_{idx}.png")
            save_image(fake_night, f"saved_images/night_{idx}.png")

        loop.set_postfix(S_real=S_reals / (idx + 1), S_fake=S_fakes / (idx + 1))
        loop.set_postfix(D_loss = D_loss)

        #Vamos rellenando las listas
        cycle_sunny_loss_list.append(cycle_sunny_loss_array)
        cycle_night_loss_list.append(cycle_night_loss_array)
        D_loss_list.append(D_loss_array)
        # G_loss_list.append(G_loss_array)

    return cycle_sunny_loss_list, cycle_night_loss_list, D_loss_list #, G_loss_list

def val_fn(
    disc_S, disc_N, gen_N, gen_S, val_loader, l1, mse
):
    disc_S.eval()
    disc_N.eval()
    gen_N.eval()
    gen_S.eval()

    S_reals = 0
    S_fakes = 0
    loop = tqdm(val_loader, leave=True)
    # Listas vacias para plotear graficas
    cycle_night_loss_list = []
    cycle_sunny_loss_list = []
    D_loss_list = []
    with torch.no_grad():

        for idx, (night, sunny) in enumerate(loop):
            night = night.to(config.DEVICE)
            sunny = sunny.to(config.DEVICE)

            # Train Discriminators S and N
            with torch.cuda.amp.autocast():
                fake_sunny = gen_S(night)
                D_S_real = disc_S(sunny)
                D_S_fake = disc_S(fake_sunny.detach())
                S_reals += D_S_real.mean().item()
                S_fakes += D_S_fake.mean().item()
                D_S_real_loss = mse(D_S_real, torch.ones_like(D_S_real))
                D_S_fake_loss = mse(D_S_fake, torch.zeros_like(D_S_fake))
                D_S_loss = D_S_real_loss + D_S_fake_loss

                fake_night = gen_N(sunny)
                D_N_real = disc_N(night)
                D_N_fake = disc_N(fake_night.detach())
                D_N_real_loss = mse(D_N_real, torch.ones_like(D_N_real))
                D_N_fake_loss = mse(D_N_fake, torch.zeros_like(D_N_fake))
                D_N_loss = D_N_real_loss + D_N_fake_loss

                # put it togethor
                D_loss = (D_S_loss + D_N_loss) / 2
                D_loss_array = D_loss.cpu().detach().numpy()

            # Train Generators S and N
            with torch.cuda.amp.autocast():
                # adversarial loss for both generators
                D_S_fake = disc_S(fake_sunny)
                D_N_fake = disc_N(fake_night)
                loss_G_S = mse(D_S_fake, torch.ones_like(D_S_fake))
                loss_G_N = mse(D_N_fake, torch.ones_like(D_N_fake))

                # cycle loss
                cycle_night = gen_N(fake_sunny)
                cycle_sunny = gen_S(fake_night)
                cycle_night_loss = l1(night, cycle_night)
                cycle_night_loss_array = cycle_night_loss.cpu().detach().numpy()
                cycle_sunny_loss = l1(sunny, cycle_sunny)
                cycle_sunny_loss_array = cycle_sunny_loss.cpu().detach().numpy()

                # identity loss (remove these for efficiency if you set lambda_identity=0)
                # identity_night = gen_N(night)
                # identity_sunny = gen_S(sunny)
                # identity_night_loss = l1(night, identity_night)
                # identity_sunny_loss = l1(sunny, identity_sunny)

                # add all togethor
                G_loss = (
                    loss_G_N
                    + loss_G_S
                    + cycle_night_loss * config.LAMBDA_CYCLE
                    + cycle_sunny_loss * config.LAMBDA_CYCLE
                    # + identity_sunny_loss * config.LAMBDA_IDENTITY
                    # + identity_night_loss * config.LAMBDA_IDENTITY
                )

            if idx % 200 == 0:
                save_image(fake_sunny * 0.5 + 0.5, f"saved_images/val_sunny_{idx}.png")
                save_image(fake_night * 0.5 + 0.5, f"saved_images/val_night_{idx}.png")

            loop.set_postfix(S_real=S_reals / (idx + 1), S_fake=S_fakes / (idx + 1))
           # loop.set_postfix(G_loss = G_loss, cycle_night_loss = cycle_night_loss, cycle_sunny_loss = cycle_sunny_loss)
            cycle_sunny_loss_list.append(cycle_sunny_loss_array)
            cycle_night_loss_list.append(cycle_night_loss_array)
            D_loss_list.append(D_loss_array)

    disc_S.train(True)
    disc_N.train(True)
    gen_N.train(True)
    gen_S.train(True)

    return cycle_sunny_loss_list, cycle_night_loss_list,D_loss_list

def main():
    disc_S = Discriminator(in_channels=3).to(config.DEVICE)
    disc_N = Discriminator(in_channels=3).to(config.DEVICE)
    gen_N = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_S = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_S.parameters()) + list(disc_N.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_N.parameters()) + list(gen_S.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_S_SN,
            gen_S,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_N_SN,
            gen_N,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISCRIM_S_SN,
            disc_S,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISCRIM_N_SN,
            disc_N,
            opt_disc,
            config.LEARNING_RATE,
        )

    dataset = SunnyNightDataset(
        root_sunny=config.TRAIN_DIR + "/sunny",
        root_night=config.TRAIN_DIR + "/night",
        transform=config.transforms,
    )
    val_dataset = SunnyNightDataset(
        root_sunny=config.VAL_DIR + "/sunny",
        root_night=config.VAL_DIR + "/night",
        transform=config.transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    #Crear listas vacias para plotear
    list_cycle_loss_sunny = []
    list_cycle_loss_night = []
    list_cycle_loss_sunny_val = []
    list_cycle_loss_night_val = []
    list_D_loss = []
    list_D_loss_val = []
    # list_G_loss = []
    list_cycle_loss_sunny_mean = []
    list_cycle_loss_night_mean = []
    list_cycle_loss_sunny_mean_val = []
    list_cycle_loss_night_mean_val = []
    list_D_loss_mean = []
    list_D_loss_mean_val = []
    # list_G_loss_mean = []
    list_epoch = []

    #Inicializar variable de cycle consistency loss que determine si un modelo es mejor
    sunny_file = open(config.MODEL_DIR_SN + "/cycle_consistency_loss_sunny_sn.txt","r")
    cycle_loss_sunny_init = float(sunny_file.read())
    sunny_file.close()
    night_file = open(config.MODEL_DIR_SN + "/cycle_consistency_loss_night_sn.txt","r")
    cycle_loss_night_init = float(night_file.read())
    night_file.close()

    for epoch in range(config.NUM_EPOCHS):
        cycle_sunny, cycle_night, D_loss = train_fn(
            disc_S,
            disc_N,
            gen_N,
            gen_S,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        list_cycle_loss_sunny.append(cycle_sunny)
        list_cycle_loss_night.append(cycle_night)
        list_D_loss.append(D_loss)
        # list_G_loss.append(G_loss)
        #Medias de las funciones de perdida para plotear
        list_cycle_loss_sunny_mean.append(np.mean(list_cycle_loss_sunny))
        list_cycle_loss_night_mean.append(np.mean(list_cycle_loss_night))
        list_D_loss_mean.append(np.mean(list_D_loss))
        # list_G_loss_mean.append(np.mean(list_G_loss))
        #Lista del numero de epochs para plotear
        list_epoch.append(epoch+1)

        #Validacion del entrenamiento
        cycle_sunny_val, cycle_night_val, D_loss_val = val_fn(disc_S, disc_N, gen_N, gen_S, val_loader, L1, mse)
        #Listas de validacion
        list_cycle_loss_sunny_val.append(cycle_sunny_val)
        list_cycle_loss_night_val.append(cycle_night_val)
        list_D_loss_val.append(D_loss_val)
        list_cycle_loss_sunny_mean_val.append(np.mean(list_cycle_loss_sunny_val))
        list_cycle_loss_night_mean_val.append(np.mean(list_cycle_loss_night_val))
        list_D_loss_mean_val.append((np.mean(list_D_loss_val)))

        if list_cycle_loss_sunny_mean_val[-1] < cycle_loss_sunny_init:
            if config.SAVE_MODEL:
                save_checkpoint(gen_S, opt_gen, filename=config.CHECKPOINT_GEN_S_SN)
                save_checkpoint(disc_S, opt_disc, filename=config.CHECKPOINT_DISCRIM_S_SN)
                sunny_file = open(config.MODEL_DIR_SN + "/cycle_consistency_loss_sunny_sn.txt", "w")
                sunny_file.write(str(list_cycle_loss_sunny_mean_val[-1]))
                sunny_file.close()
                cycle_loss_sunny_init = list_cycle_loss_sunny_mean_val[-1]

        if list_cycle_loss_night_mean_val[-1] < cycle_loss_night_init:
            if config.SAVE_MODEL:
                save_checkpoint(gen_N, opt_gen, filename=config.CHECKPOINT_GEN_N_SN)
                save_checkpoint(disc_N, opt_disc, filename=config.CHECKPOINT_DISCRIM_N_SN)
                night_file = open(config.MODEL_DIR_SN + "/cycle_consistency_loss_night_sn.txt", "w")
                night_file.write(str(list_cycle_loss_night_mean_val[-1]))
                night_file.close()
                cycle_loss_night_init = list_cycle_loss_night_mean_val[-1]

    plt.figure("Train")
    plt.plot(list_epoch,list_cycle_loss_sunny_mean, color='r', label = "Cycle Loss Sunny")
    plt.plot(list_epoch,list_cycle_loss_night_mean, color='b', label = "Cycle Loss Night")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(config.MODEL_DIR_SN + "/CycleLossTrainSN.jpg", bbox_inches = 'tight')

    plt.figure("Validation")
    plt.plot(list_epoch, list_cycle_loss_sunny_mean_val, color='r', label="Cycle Loss Sunny val")
    plt.plot(list_epoch, list_cycle_loss_night_mean_val, color='b', label="Cycle Loss Night val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(config.MODEL_DIR_SN + "/CycleLossValSN.jpg", bbox_inches = 'tight')

    plt.figure("Discriminator Loss")
    plt.plot(list_epoch, list_D_loss_mean, color='r', label="D_loss train")
    plt.plot(list_epoch, list_D_loss_mean_val, color='b', label="D_loss val")
    plt.xlabel("Epoch")
    plt.ylabel("D_Loss")
    plt.legend()
    plt.savefig(config.MODEL_DIR_SN + "/D_loss_SN.jpg", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
