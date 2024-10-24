import torch
from dataset import SunnyNightDataset
# import sys
from utils import load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
import numpy as np
# import matplotlib.pyplot as plt


def test_fn(
        disc_S, disc_N, gen_N, gen_S, test_loader, l1, mse
):
    disc_S.eval()
    disc_N.eval()
    gen_N.eval()
    gen_S.eval()

    S_reals = 0
    S_fakes = 0
    loop = tqdm(test_loader, leave=True)
    # Listas vacias para plotear graficas
    cycle_night_loss_list = []
    cycle_sunny_loss_list = []
    D_loss_list = []
    with torch.no_grad():

        for idx, (night, sunny, night_name, sunny_name) in enumerate(loop):
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

            if idx % 20 == 0:
                path = "/media/arvc/DATOS/TFG Carlos/Data/saved_images/test/SunnyNight"
                # crea la carpeta si no existe
                import os
                try:
                    os.makedirs(path)
                except OSError:
                    path = "saved_images/test/SunnyNight2"
                    print("Carpeta ya existe")

                save_image(fake_sunny * 0.5 + 0.5, path + f"/test_sunny_{night_name}_n2s.png")
                save_image(fake_night * 0.5 + 0.5, path + f"/test_night_{sunny_name}_s2n.png")
                # save_image(fake_sunny * 0.5 + 0.5, path + f"/sunny_{night_name}_n2s.png")
                # save_image(fake_night * 0.5 + 0.5, path + f"/night_{sunny_name}_s2n.png")

            loop.set_postfix(S_real=S_reals / (idx + 1), S_fake=S_fakes / (idx + 1))
            # loop.set_postfix(G_loss = G_loss, cycle_night_loss = cycle_night_loss, cycle_sunny_loss = cycle_sunny_loss)
            cycle_sunny_loss_list.append(cycle_sunny_loss_array)
            cycle_night_loss_list.append(cycle_night_loss_array)
            D_loss_list.append(D_loss_array)

    # disc_S.train(True)
    # disc_N.train(True)
    # gen_N.train(True)
    # gen_S.train(True)

    return cycle_sunny_loss_list, cycle_night_loss_list, D_loss_list


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

    # dataset = SunnyNightDataset(
    #     root_sunny=config.TRAIN_DIR + "/sunny",
    #     root_night=config.TRAIN_DIR + "/night",
    #     transform=config.transforms,
    # )
    test_dataset = SunnyNightDataset(
        root_sunny=config.TEST_DIR + "/sunny",
        root_night=config.TEST_DIR + "/night",
        transform=config.transforms, #en validacion tambien
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    # loader = DataLoader(
    #     dataset,
    #     batch_size=config.BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=config.NUM_WORKERS,
    #     pin_memory=True,
    # )
    # g_scaler = torch.cuda.amp.GradScaler()
    # d_scaler = torch.cuda.amp.GradScaler()

    # Crear listas vacias para plotear
    # list_cycle_loss_sunny = []
    # list_cycle_loss_night = []
    list_cycle_loss_sunny_test = []
    list_cycle_loss_night_test = []
    # list_D_loss = []
    list_D_loss_test = []
    # list_G_loss = []
    # list_cycle_loss_sunny_mean = []
    # list_cycle_loss_night_mean = []
    list_cycle_loss_sunny_mean_test = []
    list_cycle_loss_night_mean_test = []
    #list_D_loss_mean = []
    list_D_loss_mean_test = []
    # list_G_loss_mean = []
    #list_epoch = []

    # Inicializar variable de cycle consistency loss que determine si un modelo es mejor
    # sunny_file = open(config.MODEL_DIR_SN + "/cycle_consistency_loss_sunny_sn.txt", "r")
    # cycle_loss_sunny_init = float(sunny_file.read())
    # sunny_file.close()
    # night_file = open(config.MODEL_DIR_SN + "/cycle_consistency_loss_night_sn.txt", "r")
    # cycle_loss_night_init = float(night_file.read())
    # night_file.close()

    #for epoch in range(config.NUM_EPOCHS):
        # cycle_sunny, cycle_night, D_loss = train_fn(
        #     disc_S,
        #     disc_N,
        #     gen_N,
        #     gen_S,
        #     loader,
        #     opt_disc,
        #     opt_gen,
        #     L1,
        #     mse,
        #     d_scaler,
        #     g_scaler,
        # )

        # list_cycle_loss_sunny.append(cycle_sunny)
        # list_cycle_loss_night.append(cycle_night)
        # list_D_loss.append(D_loss)
        # list_G_loss.append(G_loss)
        # Medias de las funciones de perdida para plotear
        # list_cycle_loss_sunny_mean.append(np.mean(list_cycle_loss_sunny))
        # list_cycle_loss_night_mean.append(np.mean(list_cycle_loss_night))
        # list_D_loss_mean.append(np.mean(list_D_loss))
        # list_G_loss_mean.append(np.mean(list_G_loss))
        # Lista del numero de epochs para plotear
    #list_epoch.append(epoch + 1)

        # Testeo del entrenamiento
    cycle_sunny_test, cycle_night_test, D_loss_test = test_fn(disc_S, disc_N, gen_N, gen_S, test_loader, L1, mse)
        # Listas del test
    list_cycle_loss_sunny_test.append(cycle_sunny_test)
    list_cycle_loss_night_test.append(cycle_night_test)
    list_D_loss_test.append(D_loss_test)
    list_cycle_loss_sunny_mean_test.append(np.mean(list_cycle_loss_sunny_test))
    list_cycle_loss_night_mean_test.append(np.mean(list_cycle_loss_night_test))
    list_D_loss_mean_test.append((np.mean(list_D_loss_test)))

        # if list_cycle_loss_sunny_mean_test[-1] < cycle_loss_sunny_init:
        #     if config.SAVE_MODEL:
        #         save_checkpoint(gen_S, opt_gen, filename=config.CHECKPOINT_GEN_S_SN)
        #         save_checkpoint(disc_S, opt_disc, filename=config.CHECKPOINT_DISCRIM_S_SN)
        #         sunny_file = open(config.MODEL_DIR_SN + "/cycle_consistency_loss_sunny_sn.txt", "w")
        #         sunny_file.write(str(list_cycle_loss_sunny_mean_test[-1]))
        #         sunny_file.close()
        #         cycle_loss_sunny_init = list_cycle_loss_sunny_mean_test[-1]
        #
        # if list_cycle_loss_night_mean_test[-1] < cycle_loss_night_init:
        #     if config.SAVE_MODEL:
        #         save_checkpoint(gen_N, opt_gen, filename=config.CHECKPOINT_GEN_N_SN)
        #         save_checkpoint(disc_N, opt_disc, filename=config.CHECKPOINT_DISCRIM_N_SN)
        #         night_file = open(config.MODEL_DIR_SN + "/cycle_consistency_loss_night_sn.txt", "w")
        #         night_file.write(str(list_cycle_loss_night_mean_test[-1]))
        #         night_file.close()
        #         cycle_loss_night_init = list_cycle_loss_night_mean_test[-1]

    # plt.figure("Train")
    # plt.plot(list_epoch, list_cycle_loss_sunny_mean, color='r', label="Cycle Loss Sunny")
    # plt.plot(list_epoch, list_cycle_loss_night_mean, color='b', label="Cycle Loss Night")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.savefig(config.MODEL_DIR_SN + "/CycleLossTrainSN.jpg", bbox_inches='tight')

    # plt.figure("Test")
    # plt.plot(list_epoch, list_cycle_loss_sunny_mean_test, color='r', label="Cycle Loss Sunny test")
    # plt.plot(list_epoch, list_cycle_loss_night_mean_test, color='b', label="Cycle Loss Night test")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.savefig(config.MODEL_DIR_SN + "/CycleLossTestSN.jpg", bbox_inches='tight')
    #
    # plt.figure("Discriminator Loss")
    # #plt.plot(list_epoch, list_D_loss_mean, color='r', label="D_loss train")
    # plt.plot(list_epoch, list_D_loss_mean_test, color='b', label="D_loss test")
    # plt.xlabel("Epoch")
    # plt.ylabel("D_Loss")
    # plt.legend()
    # plt.savefig(config.MODEL_DIR_SN + "/D_loss_test_SN.jpg", bbox_inches='tight')
    # plt.show()
    print("Cycle Loss Sunny test: ", list_cycle_loss_sunny_mean_test)
    print("Cycle Loss Night test: ", list_cycle_loss_night_mean_test)
    print("D_Loss test: ", list_D_loss_mean_test)

if __name__ == "__main__":
    main()
