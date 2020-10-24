import os
import datetime
from matplotlib import pyplot as plt

import torch
from torchvision.utils import save_image

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO

from net import *
from util import *


class VAE(nn.Module):
    def __init__(self, x_ch=1, z_dim=64):
        super().__init__()
        self.encoder = Encoder(x_ch, z_dim)
        self.decoder = Decoder(x_ch, z_dim)
        self.z_dim = z_dim

    # model p(x|z)p(z)の定義
    def model(self, x):
        pyro.module('decoder', self.decoder)  # モジュールとしてデコーダーを登録
        with pyro.plate('data', len(x)):
            # 事前分布p(z)の定義
            z_loc = torch.zeros(len(x), self.z_dim, device=x.device)
            z_scale = torch.ones(len(x), self.z_dim, device=x.device)
            # 事前分布からの潜在変数zのサンプリング
            z = pyro.sample('z', dist.Normal(z_loc, z_scale).to_event(1))

            # データ分布p(x|z)p(z)の推定
            loc_img = self.decoder(z)
            # データ分布からの観測データxのサンプリング（生成）
            pyro.sample('x', dist.Bernoulli(loc_img).to_event(3), obs=x)
            return loc_img

    # guide q(z|x)の定義
    def guide(self, x):
        pyro.module('encoder', self.encoder)  # モジュールとしてエンコーダーを登録
        with pyro.plate('data', len(x)):
            # 近似事後分布q(z|x)の平均と分散の推定
            z_loc, z_scale = self.encoder(x)
            # 近似事後分布からの潜在変数zのサンプリング
            pyro.sample('z', dist.Normal(z_loc, z_scale).to_event(1))


def learn(svi, epoch, data_loader, device, train=True):
    """
    学習を実行する。train=Trueで訓練、train=Falseでテスト
    :param svi: SVI(Stochastic Variational Inference)のインスタンス
    :param epoch: 現在エポック数
    :param data_loader: データローダー。train_loader or test_loader
    :param device: GPU or CPU
    :param train: True=訓練モード False=テストモード
    :return: 損失
    """
    if train == True:
        learning_process = svi.step
    else:
        learning_process = svi.evaluate_loss

    log_interval = len(data_loader) // 10   # 進捗を表示する間隔
    if log_interval == 0:
        log_interval = 1

    loss = 0.
    for batch_idx, (x, _) in enumerate(data_loader, 1):
        x = x.to(device, non_blocking=True)

        running_loss = learning_process(x)
        loss += running_loss

        if batch_idx % log_interval == 0:
            print(f'{"train" if train == True else "test"}... [Epoch:{epoch}/Batches:{batch_idx}] loss: {running_loss / len(x):.4f}')

    loss = loss / len(data_loader.dataset)
    return loss


def reconstruct_image(encoder, decoder, x):
    """
    画像再構成を行う
    :param encoder: エンコーダー
    :param decoder: デコーダー
    :param x: 元画像Tensor
    :return: 再構成画像Tensor
    """
    with torch.no_grad():
        z_loc, z_scale = encoder(x)
        z = dist.Normal(z_loc, z_scale).sample()    # 正規分布から潜在変数をサンプリング

        x_reconst = decoder(z)
        return x_reconst


def interpolate_image(encoder, decoder, x):
    """
    潜在変数の補間によって補間した画像を生成する
    :param encoder: エンコーダー
    :param decoder: デコーダー
    :param x: 元画像Tensor
    :return: 補間画像Tensor
    """
    with torch.no_grad():
        z_loc, z_scale = encoder(x)
        z = dist.Normal(z_loc, z_scale).sample()    # 正規分布から潜在変数をサンプリング

        z_interpol = bilinear_latent(z)             # 潜在変数を双線形補間

        x_interpol = decoder(z_interpol)
        return x_interpol


def generate_image(decoder, z):
    """
    固定した潜在変数から画像を生成する
    :param decoder: デコーダー
    :param z: 潜在変数Tensor
    :return: 生成画像Tensor（潜在変数固定）
    """
    with torch.no_grad():
        x_generate = decoder(z)
        return x_generate


def sample_image(model, x_dummy):
    """
    潜在変数を事前分布からサンプリングし、その潜在変数から画像を生成する
    :param model: modelメソッド
    :param x_dummy: ダミー用Tensor
    :return: 生成画像Tensor（ランダムサンプリング）
    """
    with torch.no_grad():
        x_sample = model(x_dummy)
        return x_sample


def main(smoke_test=False):
    epochs = 2 if smoke_test == True else 50
    batch_size = 128
    seed = 0

    x_ch = 1
    z_dim = 32

    # 乱数シード初期化
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    pyro.set_rng_seed(seed)

    date_and_time = datetime.datetime.now().strftime('%Y-%m%d-%H%M')
    save_root = f'./results/pyro/{date_and_time}'
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    pyro.clear_param_store()  # Pyroのパラメーター初期化
    pyro.enable_validation(smoke_test)  # デバッグ用。NaNチェック、分布の検証、引数やサポート値チェックなど
    pyro.distributions.enable_validation(False)

    root = '/mnt/hdd/sika/Datasets'
    train_loader, test_loader = make_MNIST_loader(root, batch_size=batch_size)

    # modelメソッドとguideメソッドを持つクラスのインスタンスを作成
    vae = VAE(x_ch, z_dim).to(device)

    # 最適化アルゴリズムはPyroOptimでラッピングして使用する
    optimizer = pyro.optim.PyroOptim(torch.optim.Adam, {'lr': 1e-3})

    # SVI(Stochastic Variational Inference)のインスタンスを作成
    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

    x_fixed, _ = next(iter(test_loader))    # 固定の画像
    x_fixed = x_fixed[:8].to(device)
    z_fixed = torch.randn([64, z_dim], device=device)   # 固定の潜在変数
    x_dummy = torch.zeros(64, x_fixed.size(1), x_fixed.size(2), x_fixed.size(3), device=device)     # sample用

    train_loss_list, test_loss_list = [], []
    for epoch in range(1, epochs + 1):
        train_loss_list.append(learn(svi, epoch, train_loader, device, train=True))
        test_loss_list.append(learn(svi, epoch, test_loader, device, train=False))

        print(f'    [Epoch {epoch}] train loss {train_loss_list[-1]:.4f}')
        print(f'    [Epoch {epoch}] test  loss {test_loss_list[-1]:.4f}\n')

        # 損失値のグラフを作成し保存
        plt.plot(list(range(1, epoch+1)), train_loss_list, label='train')
        plt.plot(list(range(1, epoch + 1)), test_loss_list, label='test')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(os.path.join(save_root, 'loss.png'))
        plt.close()

        # 再構成画像
        x_reconst = reconstruct_image(vae.encoder, vae.decoder, x_fixed)
        save_image(torch.cat([x_fixed, x_reconst], dim=0), os.path.join(save_root, f'reconst_{epoch}.png'), nrow=8)

        # 補間画像
        x_interpol = interpolate_image(vae.encoder, vae.decoder, x_fixed)
        save_image(x_interpol, os.path.join(save_root, f'interpol_{epoch}.png'), nrow=8)

        # 生成画像（潜在変数固定）
        x_generate = generate_image(vae.decoder, z_fixed)
        save_image(x_generate, os.path.join(save_root, f'generate_{epoch}.png'), nrow=8)

        # 生成画像（ランダムサンプリング）
        x_sample = sample_image(vae.model, x_dummy)
        save_image(x_sample, os.path.join(save_root, f'sample_{epoch}.png'), nrow=8)


if __name__ == '__main__':
    assert pyro.__version__.startswith('1.4.0')
    main()
