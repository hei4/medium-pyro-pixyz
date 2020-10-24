import os
import datetime
from matplotlib import pyplot as plt

import torch
from torch import optim
from torchvision.utils import save_image

import pixyz
from pixyz.distributions import Normal, Bernoulli
from pixyz.losses import LogProb, KullbackLeibler, Expectation
from pixyz.models import Model

from net import *
from util import *


# 推論モデル q(z|x)の定義
class Inference(Normal):
    def __init__(self, x_ch, z_dim):
        super().__init__(cond_var=['x'], var=['z'], name='q')

        self.encoder = Encoder(x_ch, z_dim)

    def forward(self, x):
        loc, scale = self.encoder(x)

        return {'loc': loc, 'scale': scale}


# 生成モデル p(x|z)の定義
class Generator(Bernoulli):
    def __init__(self, x_ch, z_dim):
        super().__init__(cond_var=['z'], var=['x'], name='p')

        self.decoder = Decoder(x_ch, z_dim)

    def forward(self, z):
        probs = self.decoder(z)

        return {'probs': probs}


def learn(model, epoch, data_loader, device, train=True):
    """
    学習を実行する。train=Trueで訓練、train=Falseでテスト
    :param model: Model APIのインスタンス
    :param epoch: 現在エポック数
    :param data_loader: データローダー。train_loader or test_loader
    :param device: GPU or CPU
    :param train: True=訓練モード False=テストモード
    :return: 損失
    """
    if train == True:
        learning_process = model.train
    else:
        learning_process = model.test

    log_interval = len(data_loader) // 10   # 進捗を表示する間隔
    if log_interval == 0:
        log_interval = 1

    loss = 0
    for batch_idx, (x, _) in enumerate(data_loader, 1):
        x = x.to(device, non_blocking=True)

        running_loss = learning_process({'x': x})
        loss += running_loss

        if batch_idx % log_interval == 0:
            print(f'{"train" if train == True else "test"}... [Epoch:{epoch}/Batches:{batch_idx}] loss: {running_loss / len(x):.4f}')

    loss = loss * data_loader.batch_size / len(data_loader.dataset)

    return loss.item()


def reconstruct_image(p, q, x):
    """
    画像再構成を行う
    :param p: 生成モデル
    :param q: 推論モデル
    :param x: 元画像Tensor
    :return: 再構成画像Tensor
    """
    with torch.no_grad():
        z = q.sample({'x': x}, return_all=False)

        x_reconst = p.sample_mean(z)
        return x_reconst


def interpolate_image(p, q, x):
    """
    潜在変数の補間によって補間した画像を生成する
    :param p: 生成モデル
    :param q: 推論モデル
    :param x: 元画像Tensor
    :return: 補間画像Tensor
    """
    with torch.no_grad():
        z = q.sample({'x': x}, return_all=False)

        z_interpol = bilinear_latent(z['z'])    # 潜在変数を双線形補間

        x_interpol = p.sample_mean({'z': z_interpol})
        return x_interpol


def generate_image(p, z):
    """
    固定した潜在変数から画像を生成する
    :param p: 生成モデル
    :param z: 潜在変数Tensor
    :return: 生成画像Tensor（潜在変数固定）
    """
    with torch.no_grad():
        x_generate = p.sample_mean({'z': z})

        return x_generate


def sample_image(p_prior, p):
    """
    潜在変数を事前分布からサンプリングし、その潜在変数から画像を生成する
    :param p_prior: 潜在変数の事前分布
    :param p: 生成モデル
    :return: 生成画像Tensor（ランダムサンプリング）
    """
    with torch.no_grad():
        z_sample = p_prior.sample(batch_n=64)
        x_sample = p.sample_mean(z_sample)
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

    date_and_time = datetime.datetime.now().strftime('%Y-%m%d-%H%M')
    save_root = f'./results/pixyz/{date_and_time}'
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    root = '/mnt/hdd/sika/Datasets'
    train_loader, test_loader = make_MNIST_loader(root, batch_size=batch_size)

    # 生成モデルと推論モデルの生成
    p = Generator(x_ch, z_dim).to(device)
    q = Inference(x_ch, z_dim).to(device)

    # 潜在変数の事前分布の規定
    p_prior = Normal(loc=torch.tensor(0.), scale=torch.tensor(1.),
                     var=['z'], features_shape=[z_dim], name='p_{prior}').to(device)

    # 損失関数の定義
    loss = (KullbackLeibler(q, p_prior) - Expectation(q, LogProb(p))).mean()

    # Model APIの設定
    model = Model(loss=loss, distributions=[p, q],
                  optimizer=optim.Adam, optimizer_params={"lr": 1e-3})

    x_fixed, _ = next(iter(test_loader))
    x_fixed = x_fixed[:8].to(device)
    z_fixed = p_prior.sample(batch_n=64)['z']

    train_loss_list, test_loss_list = [], []
    for epoch in range(1, epochs + 1):
        train_loss_list.append(learn(model, epoch, train_loader, device, train=True))
        test_loss_list.append(learn(model, epoch, test_loader, device, train=False))

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
        x_reconst = reconstruct_image(p, q, x_fixed)
        save_image(torch.cat([x_fixed, x_reconst], dim=0), os.path.join(save_root, f'reconst_{epoch}.png'), nrow=8)

        # 補間画像
        x_interpol = interpolate_image(p, q, x_fixed)
        save_image(x_interpol, os.path.join(save_root, f'interpol_{epoch}.png'), nrow=8)

        # 生成画像（潜在変数固定）
        x_generate = generate_image(p, z_fixed)
        save_image(x_generate, os.path.join(save_root, f'generate_{epoch}.png'), nrow=8)

        # 生成画像（ランダムサンプリング）
        x_sample = sample_image(p_prior, p)
        save_image(x_sample, os.path.join(save_root, f'sample_{epoch}.png'), nrow=8)


if __name__ == '__main__':
    assert pixyz.__version__.startswith('0.2.0')
    main()