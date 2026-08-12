from torch import nn
from torch.nn import functional as F
import torch


class MultimodalLoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.gamma = args['base']['gamma']
        self.sigma = args['base']['sigma']
        self.kl = args['base']['kl']
        self.target_alpha = args['base']['alpha_distill']


        self.MSE_Fn = nn.MSELoss()


        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, out, label):
        raw_labels = label['sentiment_labels']  # [B, 1]

        l_sp = self.MSE_Fn(out['sentiment_preds'], raw_labels)

        l_rec = (
            self.MSE_Fn(out['rec_feats'], out['complete_feats'])
            if out['rec_feats'] is not None and out['complete_feats'] is not None
            else torch.tensor(0.0, device=raw_labels.device)
        )

        l_sr = out['sr_loss']

        l_align = out['align_loss']

        l_feat_distill = torch.tensor(0.0, device=raw_labels.device)

        if out.get('teacher_proxy_q') is not None:

            l_feat_distill = self.MSE_Fn(
                out['student_proxy_q'],
                out['teacher_proxy_q'].detach()
            )

        l_weight_distill = torch.tensor(0.0, device=label['sentiment_labels'].device)
        if out.get('teacher_weight') is not None and out.get('student_weight') is not None:
            student_w = out['student_weight'].squeeze(-1).squeeze(-1).t()  # [Batch, 3]
            teacher_w = out['teacher_weight'].squeeze(-1).squeeze(-1).t().detach()

            log_student_w = torch.log(student_w + 1e-8)
            l_weight_distill = torch.nn.functional.kl_div(
                log_student_w,
                teacher_w,
                reduction='batchmean'
            )

        current_alpha = 0.0
        if self.current_epoch > 5:
            current_alpha = self.target_alpha * min(1.0, (self.current_epoch - 5) / 5.0)

        loss = (
                self.sigma * l_sp + self.gamma * l_rec
                + self.kl * l_sr  + current_alpha * (l_feat_distill+l_weight_distill)
        )

        return {
            'loss': loss,
            'l_sp': l_sp,
            'l_rec': l_rec,
            'l_kl': l_sr,
            'l_feat_distill': l_feat_distill+l_weight_distill,
            'l_align': l_align,

        }