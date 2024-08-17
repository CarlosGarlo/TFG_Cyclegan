"""
Training for CycleGAN

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-05: Initial coding
* 2022-12-21: Small revision of code, checked that it works with latest PyTorch version
"""

import torch
from dataset import NightCloudyDataset
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
    disc_N, disc_C, gen_C, gen_N, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    N_reals = 0
    N_fakes = 0
    loop = tqdm(loader, leave=True)
    # Listas vacias para plotear graficas
    cycle_cloudy_loss_list = []
    cycle_night_loss_list = []
    # D_loss_list = []
    # G_loss_list = []

    for idx, (cloudy, night) in enumerate(loop):
        cloudy = cloudy.to(config.DEVICE)
        night = night.to(config.DEVICE)

        # Train Discriminators S and N
        with torch.cuda.amp.autocast():
            fake_night = gen_N(cloudy)
            D_N_real = disc_N(night)
            D_N_fake = disc_N(fake_night.detach())
            N_reals += D_N_real.mean().item()
            N_fakes += D_N_fake.mean().item()
            D_N_real_loss = mse(D_N_real, torch.ones_like(D_N_real))
            D_N_fake_loss = mse(D_N_fake, torch.zeros_like(D_N_fake))
            D_N_loss = D_N_real_loss + D_N_fake_loss

            fake_cloudy = gen_C(night)
            D_C_real = disc_C(cloudy)
            D_C_fake = disc_C(fake_cloudy.detach())
            D_C_real_loss = mse(D_C_real, torch.ones_like(D_C_real))
            D_C_fake_loss = mse(D_C_fake, torch.zeros_like(D_C_fake))
            D_C_loss = D_C_real_loss + D_C_fake_loss

            # put it togethor
            D_loss = (D_N_loss + D_C_loss) / 2
           # D_loss_array = D_loss.cpu().detach().numpy()

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators S and N
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_N_fake = disc_N(fake_night)
            D_C_fake = disc_C(fake_cloudy)
            loss_G_N = mse(D_N_fake, torch.ones_like(D_N_fake))
            loss_G_C = mse(D_C_fake, torch.ones_like(D_C_fake))

            # cycle loss
            cycle_cloudy = gen_C(fake_night)
            cycle_night = gen_N(fake_cloudy)
            cycle_cloudy_loss = l1(cloudy, cycle_cloudy)
            cycle_cloudy_loss_array = cycle_cloudy_loss.cpu().detach().numpy()
            cycle_night_loss = l1(night, cycle_night)
            cycle_night_loss_array = cycle_night_loss.cpu().detach().numpy()
            # identity loss (remove these for efficiency if you set lambda_identity=0)
            # identity_cloudy = gen_C(cloudy)
            # identity_night = gen_N(night)
            # identity_cloudy_loss = l1(cloudy, identity_cloudy)
            # identity_night_loss = l1(night, identity_night)

            # add all togethor
            G_loss = (
                loss_G_C
                + loss_G_N
                + cycle_cloudy_loss * config.LAMBDA_CYCLE
                + cycle_night_loss * config.LAMBDA_CYCLE
                # + identity_night_loss * config.LAMBDA_IDENTITY
                # + identity_cloudy_loss * config.LAMBDA_IDENTITY
            )
           # G_loss_array = G_loss.detach().to('cpu').numpy()

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 2 == 0:
            save_image(fake_night * 0.5 + 0.5, f"saved_images/night_{idx}.png")
            save_image(fake_cloudy * 0.5 + 0.5, f"saved_images/cloudy_{idx}.png")

        loop.set_postfix(N_real=N_reals / (idx + 1), N_fake=N_fakes / (idx + 1))
        # loop.set_postfix(G_loss = G_loss, cycle_cloudy_loss = cycle_cloudy_loss, cycle_night_loss = cycle_night_loss, D_loss = D_loss)

        #Vamos rellenando las listas
        cycle_night_loss_list.append(cycle_night_loss_array)
        cycle_cloudy_loss_list.append(cycle_cloudy_loss_array)
        # D_loss_list.append(D_loss_array)
        # G_loss_list.append(G_loss_array)

    return cycle_night_loss_list, cycle_cloudy_loss_list #, D_loss_list, G_loss_list

def val_fn(
    disc_N, disc_C, gen_C, gen_N, val_loader, l1, mse
):
    disc_N.eval()
    disc_C.eval()
    gen_C.eval()
    gen_N.eval()

    S_reals = 0
    S_fakes = 0
    loop = tqdm(val_loader, leave=True)
    # Listas vacias para plotear graficas
    cycle_cloudy_loss_list = []
    cycle_night_loss_list = []
    with torch.no_grad():

        for idx, (cloudy, night) in enumerate(loop):
            cloudy = cloudy.to(config.DEVICE)
            night = night.to(config.DEVICE)

            # Train Discriminators S and N
            with torch.cuda.amp.autocast():
                fake_night = gen_N(cloudy)
                D_N_real = disc_N(night)
                D_N_fake = disc_N(fake_night.detach())
                S_reals += D_N_real.mean().item()
                S_fakes += D_N_fake.mean().item()
                D_N_real_loss = mse(D_N_real, torch.ones_like(D_N_real))
                D_N_fake_loss = mse(D_N_fake, torch.zeros_like(D_N_fake))
                D_N_loss = D_N_real_loss + D_N_fake_loss

                fake_cloudy = gen_C(night)
                D_C_real = disc_C(cloudy)
                D_C_fake = disc_C(fake_cloudy.detach())
                D_C_real_loss = mse(D_C_real, torch.ones_like(D_C_real))
                D_C_fake_loss = mse(D_C_fake, torch.zeros_like(D_C_fake))
                D_C_loss = D_C_real_loss + D_C_fake_loss

                # put it togethor
                D_loss = (D_N_loss + D_C_loss) / 2


            # Train Generators S and N
            with torch.cuda.amp.autocast():
                # adversarial loss for both generators
                D_N_fake = disc_N(fake_night)
                D_C_fake = disc_C(fake_cloudy)
                loss_G_N = mse(D_N_fake, torch.ones_like(D_N_fake))
                loss_G_C = mse(D_C_fake, torch.ones_like(D_C_fake))

                # cycle loss
                cycle_cloudy = gen_C(fake_night)
                cycle_night = gen_N(fake_cloudy)
                cycle_cloudy_loss = l1(cloudy, cycle_cloudy)
                cycle_cloudy_loss_array = cycle_cloudy_loss.cpu().detach().numpy()
                cycle_night_loss = l1(night, cycle_night)
                cycle_night_loss_array = cycle_night_loss.cpu().detach().numpy()

                # identity loss (remove these for efficiency if you set lambda_identity=0)
                # identity_cloudy = gen_C(cloudy)
                # identity_night = gen_N(night)
                # identity_cloudy_loss = l1(cloudy, identity_cloudy)
                # identity_night_loss = l1(night, identity_night)

                # add all togethor
                G_loss = (
                    loss_G_C
                    + loss_G_N
                    + cycle_cloudy_loss * config.LAMBDA_CYCLE
                    + cycle_night_loss * config.LAMBDA_CYCLE
                    # + identity_night_loss * config.LAMBDA_IDENTITY
                    # + identity_cloudy_loss * config.LAMBDA_IDENTITY
                )

            if idx % 2 == 0:
                save_image(fake_night * 0.5 + 0.5, f"saved_images/val_night_{idx}.png")
                save_image(fake_cloudy * 0.5 + 0.5, f"saved_images/val_cloudy_{idx}.png")

            loop.set_postfix(S_real=S_reals / (idx + 1), S_fake=S_fakes / (idx + 1))
           # loop.set_postfix(G_loss = G_loss, cycle_cloudy_loss = cycle_cloudy_loss, cycle_night_loss = cycle_night_loss)
            cycle_night_loss_list.append(cycle_night_loss_array)
            cycle_cloudy_loss_list.append(cycle_cloudy_loss_array)

    disc_N.train(True)
    disc_C.train(True)
    gen_C.train(True)
    gen_N.train(True)

    return cycle_night_loss_list, cycle_cloudy_loss_list

def main():
    disc_N = Discriminator(in_channels=3).to(config.DEVICE)
    disc_C = Discriminator(in_channels=3).to(config.DEVICE)
    gen_C = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_N = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_N.parameters()) + list(disc_C.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_C.parameters()) + list(gen_N.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_N_NC,
            gen_N,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_C_NC,
            gen_C,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISCRIM_N_NC,
            disc_N,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISCRIM_C_NC,
            disc_C,
            opt_disc,
            config.LEARNING_RATE,
        )

    dataset = NightCloudyDataset(
        root_night=config.TRAIN_DIR + "/night",
        root_cloudy=config.TRAIN_DIR + "/cloudy",
        transform=config.transforms,
    )
    val_dataset = NightCloudyDataset(
        root_night=config.VAL_DIR + "/night",
        root_cloudy=config.VAL_DIR + "/cloudy",
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
    list_cycle_loss_night = []
    list_cycle_loss_cloudy = []
    list_cycle_loss_night_val = []
    list_cycle_loss_cloudy_val = []
    # list_D_loss = []
    # list_G_loss = []
    list_cycle_loss_night_mean = []
    list_cycle_loss_cloudy_mean = []
    list_cycle_loss_night_mean_val = []
    list_cycle_loss_cloudy_mean_val = []
    # list_D_loss_mean = []
    # list_G_loss_mean = []
    list_epoch = []

    #Inicializar variable de cycle consistency loss que determine si un modelo es mejor
    night_file = open(config.MODEL_DIR_NC + "/cycle_consistency_loss_night_nc.txt","r")
    cycle_loss_night_init = float(night_file.read())
    night_file.close()
    cloudy_file = open(config.MODEL_DIR_NC + "/cycle_consistency_loss_cloudy_nc.txt","r")
    cycle_loss_cloudy_init = float(cloudy_file.read())
    cloudy_file.close()

    for epoch in range(config.NUM_EPOCHS):
        cycle_night, cycle_cloudy = train_fn(
            disc_N,
            disc_C,
            gen_C,
            gen_N,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        list_cycle_loss_night.append(cycle_night)
        list_cycle_loss_cloudy.append(cycle_cloudy)
        # list_D_loss.append(D_loss)
        # list_G_loss.append(G_loss)
        #Medias de las funciones de perdida para plotear
        list_cycle_loss_night_mean.append(np.mean(list_cycle_loss_night))
        list_cycle_loss_cloudy_mean.append(np.mean(list_cycle_loss_cloudy))
        # list_D_loss_mean.append(np.mean(list_D_loss))
        # list_G_loss_mean.append(np.mean(list_G_loss))
        #Lista del numero de epochs para plotear
        list_epoch.append(epoch+1)

        #Validacion del entrenamiento
        cycle_night_val, cycle_cloudy_val = val_fn(disc_N, disc_C, gen_C, gen_N, val_loader, L1, mse)
        #Listas de validacion
        list_cycle_loss_night_val.append(cycle_night_val)
        list_cycle_loss_cloudy_val.append(cycle_cloudy_val)
        list_cycle_loss_night_mean_val.append(np.mean(list_cycle_loss_night_val))
        list_cycle_loss_cloudy_mean_val.append(np.mean(list_cycle_loss_cloudy_val))

        if list_cycle_loss_night_mean_val[-1] < cycle_loss_night_init:
            if config.SAVE_MODEL:
                save_checkpoint(gen_N, opt_gen, filename=config.CHECKPOINT_GEN_N_NC)
                save_checkpoint(disc_N, opt_disc, filename=config.CHECKPOINT_DISCRIM_N_NC)
                night_file = open(config.MODEL_DIR_NC + "/cycle_consistency_loss_night_nc.txt", "w")
                night_file.write(str(list_cycle_loss_night_mean_val[-1]))
                night_file.close()
                cycle_loss_night_init = list_cycle_loss_night_mean_val[-1]

        if list_cycle_loss_cloudy_mean_val[-1] < cycle_loss_cloudy_init:
            if config.SAVE_MODEL:
                save_checkpoint(gen_C, opt_gen, filename=config.CHECKPOINT_GEN_C_NC)
                save_checkpoint(disc_C, opt_disc, filename=config.CHECKPOINT_DISCRIM_C_NC)
                cloudy_file = open(config.MODEL_DIR_NC + "/cycle_consistency_loss_cloudy_nc.txt", "w")
                cloudy_file.write(str(list_cycle_loss_cloudy_mean_val[-1]))
                cloudy_file.close()
                cycle_loss_cloudy_init = list_cycle_loss_cloudy_mean_val[-1]

    plt.figure("Train")
    plt.plot(list_epoch,list_cycle_loss_night_mean, color='r', label = "Cycle Loss night")
    plt.plot(list_epoch,list_cycle_loss_cloudy_mean, color='b', label = "Cycle Loss cloudy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.figure("Validation")
    plt.plot(list_epoch, list_cycle_loss_night_mean_val, color='r', label="Cycle Loss night val")
    plt.plot(list_epoch, list_cycle_loss_cloudy_mean_val, color='b', label="Cycle Loss cloudy val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
