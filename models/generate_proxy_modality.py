import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_layers import Transformer

class FSQ(nn.Module):
    def __init__(self, levels: list[int]):
        super().__init__()
        self.register_buffer('levels', torch.tensor(levels))
        self.register_buffer('basis', torch.tensor([1] + [p for p in levels[:-1]]).cumprod(0))
        self.register_buffer('half_width',
                             torch.tensor([l // 2 for l in levels], dtype=torch.float32))
    def bound(self, z, eps=1e-3):
        """Bound input to [ -0.5, levels - 0.5]"""
        half_l = (self.levels - 1) * (1 - eps) / 2
        offset = torch.where(self.levels % 2 == 1, 0.0, 0.5)
        shift = (offset / half_l).tan()
        return (z + shift).tanh() * half_l - offset

        # return F.softsign(z + shift) * half_l - offset

    def quantize(self, z):
        """Quantize with Straight-Through Estimator (STE)"""
        # 1. Bounding
        z_bounded = self.bound(z)
        renormalized_z = z_bounded

        # 2. Quantization (Round)
        z_q = torch.round(renormalized_z)
        z_q = renormalized_z + (z_q - renormalized_z).detach()  # STE

        # 3. Normalize back to [-1, 1] range for Decoder
        z_out = z_q / self.half_width

        return z_out


    def forward(self, z):
        return self.quantize(z)

class VariationalEncoder(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, latent_dim):
        super(VariationalEncoder, self).__init__()

        self.encoder = nn.Sequential(
            Transformer(num_frames=args['model']['vae']['input_length'],
                        save_hidden=False,
                        token_len=None,
                        dim=args['model']['vae']['input_dim'],
                        depth=args['model']['vae']['depth'],
                        heads=args['model']['vae']['heads'],
                        mlp_dim=args['model']['vae']['hidden_dim'])
        )

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_z = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        memory = self.encoder(h)
        z = self.fc_z(memory)  # Direct projection
        return z


class Decoder(nn.Module):
    def __init__(self, args, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)

        self.decoder = nn.Sequential(
            Transformer(num_frames=args['model']['vae']['input_length'],
                        save_hidden=False,
                        token_len=None,
                        dim=args['model']['vae']['input_dim'],
                        depth=args['model']['vae']['depth'],
                        heads=args['model']['vae']['heads'],
                        mlp_dim=args['model']['vae']['hidden_dim'])
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        output = self.decoder(h)
        output = self.fc_out(output)
        return output


class VAE(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder(args, input_dim, hidden_dim, latent_dim)

        level_int = args['base']['level']
        levels = [level_int] * latent_dim
        self.fsq = FSQ(levels)

        self.decoder = Decoder(args, latent_dim, hidden_dim, input_dim)

    def encode(self, x):
        z_continuous = self.encoder(x)
        z_q = self.fsq(z_continuous)

        return z_q, z_continuous
    def forward(self, x):
        z_continuous = self.encoder(x)

        z_q = self.fsq(z_continuous)

        x_recon = self.decoder(z_q)
        return x_recon, z_q, z_continuous


def recon_loss(x, x_recon):
    mse_loss = nn.MSELoss()
    return mse_loss(x_recon, x)

class Generate_Proxy_Modality(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, latent_dim):
        super(Generate_Proxy_Modality, self).__init__()
        self.text_VAE = VAE(args, input_dim, hidden_dim, latent_dim)
        self.video_VAE = VAE(args, input_dim, hidden_dim, latent_dim)
        self.audio_VAE = VAE(args, input_dim, hidden_dim, latent_dim)


        self.weight_learner_t= nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.weight_learner_a = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.weight_learner_v= nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, text, video, audio, c_text, c_vision, c_audio, presence_rates=None):
        # 1. Forward Pass (FSQ)
        if self.training:
            t_recon, z_t,  z_c_t = self.text_VAE(text)
            v_recon, z_v,  z_c_v = self.video_VAE(video)
            a_recon, z_a,  z_c_a = self.audio_VAE(audio)
            # 2. Reconstruction Loss ONLY
            if (c_text is not None) and (c_vision is not None) and (c_audio is not None):
                loss_t = recon_loss(c_text, t_recon)
                loss_v = recon_loss(c_vision, v_recon)
                loss_a = recon_loss(c_audio, a_recon)
            else:
                loss_t = recon_loss(text, t_recon)
                loss_v = recon_loss(video, v_recon)
                loss_a = recon_loss(audio, a_recon)
            sr_loss = (loss_t + loss_v + loss_a ) / 3
        else:
            z_t, z_c_t = self.text_VAE.encode(text)
            z_v, z_c_v = self.video_VAE.encode(video)
            z_a, z_c_a = self.audio_VAE.encode(audio)

            sr_loss = torch.zeros(
                (),
                device=text.device,
                dtype=text.dtype
            )


        score_t = self.weight_learner_t(z_t.mean(dim=1))
        score_v = self.weight_learner_v(z_v.mean(dim=1))
        score_a = self.weight_learner_a(z_a.mean(dim=1))
        # # [Batch, 3]_t
        scores = torch.cat([score_t, score_v, score_a], dim=1)

        align_loss = 0.0
        if self.training and presence_rates is not None:
            pred_rates = torch.sigmoid(scores)
            mse_fn = nn.MSELoss()
            align_loss = mse_fn(pred_rates, presence_rates)
        weight_m = F.softmax(scores, dim=1).t().unsqueeze(-1).unsqueeze(-1)

        # z: [Batch, Seq, Dim] -> stack -> [3, Batch, Seq, Dim]
        mu_i_m = torch.stack([z_t, z_v, z_a], dim=0)
        proxy_m = torch.sum(weight_m * mu_i_m, dim=0)

        u_i_m = torch.stack([z_c_t, z_c_v, z_c_a], dim=0)
        proxy_c = torch.sum(weight_m * u_i_m, dim=0)

        return sr_loss, proxy_m, weight_m,proxy_c, align_loss
