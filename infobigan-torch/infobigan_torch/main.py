import infobigan_torch
ibg = infobigan_torch.InfoBiGAN(channels=(1, 1024, 512, 256), padding=0, kernel_size=(28, 1, 1), latent_dim=50,
                                manifest_dim=28)
# ibg = infobigan_torch.InfoBiGAN(channels=(1, 8, 8, 8), padding=0, kernel_size=28, latent_dim=50)
# mnist = infobigan_torch.load_mnist()
ibg_t = infobigan_torch.InfoBiGANTrainer(mnist, ibg)
ibg_t.train()