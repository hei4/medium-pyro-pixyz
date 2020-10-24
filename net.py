from torch import nn


class Encoder(nn.Module):
    def __init__(self, x_ch, z_dim):
        super().__init__()

        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(x_ch, 32, kernel_size=3, stride=1, padding=1),    # [N, 32, 28, 28]
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)                                 # [N, 32, 14, 14]
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),      # [N, 64, 14, 14]
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)                                 # [N, 64, 7, 7]
            ),
            nn.Sequential(
                nn.Conv2d(64, 512, kernel_size=7, stride=1, padding=0),     # [N, 512, 1, 1]
                nn.ReLU(),
                nn.Flatten()                                                # [N, 512]
            )
        ])

        self.loc_layer = nn.Linear(512, z_dim)  # [N, z_dim]

        self.scale_layer = nn.Sequential(
            nn.Linear(512, z_dim),              # [N, z_dim]
            nn.Softplus()
        )

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        return self.loc_layer(x), self.scale_layer(x)



class Decoder(nn.Module):
    def __init__(self, x_ch, z_dim):
        super().__init__()
        self.z_dim = z_dim

        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(z_dim, 512, kernel_size=1, stride=1, padding=0),          # [N, 512, 1, 1]
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(512, 64, kernel_size=7, stride=1, padding=0),    # [N, 64, 7, 7]
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),     # [N, 32, 14, 14]
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(32, x_ch, kernel_size=4, stride=2, padding=1),   # [N, x_ch, 28, 28]
                nn.Sigmoid(),
            )
        ])

    def forward(self, z):
        z = z.reshape(-1, self.z_dim, 1, 1)     # [N, z_dim, 1, 1]
        for layer in self.conv_layers:
            z = layer(z)

        return z
