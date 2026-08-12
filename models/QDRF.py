import torch
from torch import nn
import torch.nn.functional as F
from .basic_layers import Transformer, CrossmodalEncoder
from .bert import BertTextEncoder
from .generate_proxy_modality import Generate_Proxy_Modality

class QDRF(nn.Module):
    def __init__(self, args):
        super(QDRF, self).__init__()

        self.bertmodel = BertTextEncoder(use_finetune=True,
                                         transformers=args['model']['feature_extractor']['transformers'],
                                         pretrained=args['model']['feature_extractor']['bert_pretrained'])

        self.proj_l = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][0],
                      args['model']['feature_extractor']['hidden_dims'][0]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][0],
                        save_hidden=False,
                        token_len=args['model']['feature_extractor']['token_length'][0],
                        dim=args['model']['feature_extractor']['hidden_dims'][0],
                        depth=args['model']['feature_extractor']['depth'],
                        heads=args['model']['feature_extractor']['heads'],
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][0])
        )

        self.proj_a = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][2],
                      args['model']['feature_extractor']['hidden_dims'][2]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][2],
                        save_hidden=False,
                        token_len=args['model']['feature_extractor']['token_length'][2],
                        dim=args['model']['feature_extractor']['hidden_dims'][2],
                        depth=args['model']['feature_extractor']['depth'],
                        heads=args['model']['feature_extractor']['heads'],
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][2])
        )

        self.proj_v = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][1],
                      args['model']['feature_extractor']['hidden_dims'][1]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][1],
                        save_hidden=False,
                        token_len=args['model']['feature_extractor']['token_length'][1],
                        dim=args['model']['feature_extractor']['hidden_dims'][1],
                        depth=args['model']['feature_extractor']['depth'],
                        heads=args['model']['feature_extractor']['heads'],
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][1])
        )

        self.generate_proxy_modality = Generate_Proxy_Modality(args, args['model']['generate_proxy']['input_dim'],
                                                               args['model']['generate_proxy']['hidden_dim'],
                                                               args['model']['generate_proxy']['out_dim'])

        self.reconstructor = nn.ModuleList([
            Transformer(num_frames=args['model']['reconstructor']['input_length'],
                        save_hidden=False,
                        token_len=None,
                        dim=args['model']['reconstructor']['input_dim'],
                        depth=args['model']['reconstructor']['depth'],
                        heads=args['model']['reconstructor']['heads'],
                        mlp_dim=args['model']['reconstructor']['hidden_dim']) for _ in range(3)
        ])

        self.crossmodal_encoder = CrossmodalEncoder(proxy_dim=args['model']['crossmodal_encoder']['proxy_dim'],
                                                    text_dim=args['model']['crossmodal_encoder']['hidden_dims'][0],
                                                    audio_dim=args['model']['crossmodal_encoder']['hidden_dims'][2],
                                                    video_dim=args['model']['crossmodal_encoder']['hidden_dims'][1],
                                                    embed_dim=args['model']['crossmodal_encoder']['embed_dim'],
                                                    num_layers=args['model']['crossmodal_encoder']['num_layers'],
                                                    attn_dropout=args['model']['crossmodal_encoder']['attn_dropout'])

        self.fc1 = nn.Linear(args['model']['regression']['input_dim'], args['model']['regression']['hidden_dim'])
        self.fc2 = nn.Linear(args['model']['regression']['hidden_dim'], args['model']['regression']['out_dim'])
        self.dropout = nn.Dropout(args['model']['regression']['attn_dropout'])

        self.teacher = self._build_teacher(args)
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.ema_decay = args['base'].get('ema_decay', 0.999)

    def _build_teacher(self, args):
        teacher = TeacherNet(args)
        student_sd = {k: v for k, v in self.state_dict().items()
                      if not k.startswith('teacher.')}
        teacher_sd = {}
        for t_name, _ in teacher.named_parameters():
            if t_name in student_sd:
                teacher_sd[t_name] = student_sd[t_name].clone()
        teacher.load_state_dict(teacher_sd, strict=False)
        return teacher
    @torch.no_grad()
    def update_ema(self):
        student_sd = self.state_dict()
        for t_name, t_param in self.teacher.named_parameters():
            s_key = t_name
            if s_key in student_sd:
                t_param.data.mul_(self.ema_decay).add_(
                    student_sd[s_key].data, alpha=1.0 - self.ema_decay
                )

    def _get_student_params(self):
        for name, param in self.named_parameters():
            if not name.startswith('teacher.'):
                yield param


    def predict(self, x):
        output = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return output

    def forward(self, complete_input, incomplete_input, masks=None):
        vision, audio, language = complete_input
        vision_m, audio_m, language_m = incomplete_input
        presence_rates = None
        if self.training and masks is not None:
            mask_v, mask_a, mask_l = masks
            presence_v = 1.0 - mask_v.float().mean(dim=1, keepdim=True)
            presence_a = 1.0 - mask_a.float().mean(dim=1, keepdim=True)
            presence_l = 1.0 - mask_l.float().mean(dim=1, keepdim=True)

            presence_rates = torch.cat([presence_l, presence_v, presence_a], dim=1)

        h_1_v = self.proj_v(vision_m)[:, :8]
        h_1_a = self.proj_a(audio_m)[:, :8]
        h_1_l = self.proj_l(self.bertmodel(language_m))[:, :8]

        complete_language_feat, complete_vision_feat, complete_audio_feat = None, None, None
        if (vision is not None) and (audio is not None) and (language is not None):
            with torch.no_grad():
                complete_language_feat = self.proj_l(self.bertmodel(language))[:, :8]
                complete_vision_feat = self.proj_v(vision)[:, :8]
                complete_audio_feat = self.proj_a(audio)[:, :8]

        sr_loss, proxy_m, weight_t_v_a, proxy_c,  align_loss= self.generate_proxy_modality(
            h_1_l, h_1_v, h_1_a, complete_language_feat,complete_vision_feat, complete_audio_feat ,presence_rates
        )

        feat = self.crossmodal_encoder(proxy_m, h_1_l, h_1_a, h_1_v, weight_t_v_a)
        output = self.predict(torch.mean(feat, dim=1))

        # ---- EMA ----
        teacher_output = None
        teacher_proxy_q = None
        if self.training and (vision is not None):
            with torch.no_grad():
                teacher_output, teacher_proxy_q,teacher_weight= self.teacher(complete_input)

        # ---- Reconstruction----
        rec_feats, complete_feats = None, None
        if vision is not None:
            complete_language_feat = self.proj_l(self.bertmodel(language))[:, :8]
            complete_vision_feat = self.proj_v(vision)[:, :8]
            complete_audio_feat = self.proj_a(audio)[:, :8]
            rec_feat_a = self.reconstructor[0](h_1_a)[:, :8]
            rec_feat_v = self.reconstructor[1](h_1_v)[:, :8]
            rec_feat_l = self.reconstructor[2](h_1_l)[:, :8]
            rec_feats = torch.cat([rec_feat_a, rec_feat_v, rec_feat_l], dim=1)
            complete_feats = torch.cat(
                [complete_audio_feat, complete_vision_feat, complete_language_feat], dim=1
            )

        return {
            'sentiment_preds': output,
            'rec_feats': rec_feats,
            'complete_feats': complete_feats,
            'sr_loss': sr_loss,
            'teacher_preds': teacher_output,
            'student_proxy_q': proxy_m,
            'student_proxy_c': proxy_c,
            'teacher_proxy_q': teacher_proxy_q,
            'teacher_proxy_c': teacher_proxy_q,
            'student_weight': weight_t_v_a,
            'teacher_weight': teacher_weight if 'teacher_weight' in locals() else None,
            'align_loss': align_loss
        }

def build_model(args):
    return QDRF(args)


class TeacherNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bertmodel = BertTextEncoder(
            use_finetune=True,
            transformers=args['model']['feature_extractor']['transformers'],
            pretrained=args['model']['feature_extractor']['bert_pretrained']
        )
        self.proj_l = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][0],
                      args['model']['feature_extractor']['hidden_dims'][0]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][0],
                        save_hidden=False,
                        token_len=args['model']['feature_extractor']['token_length'][0],
                        dim=args['model']['feature_extractor']['hidden_dims'][0],
                        depth=args['model']['feature_extractor']['depth'],
                        heads=args['model']['feature_extractor']['heads'],
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][0])
        )
        self.proj_a = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][2],
                      args['model']['feature_extractor']['hidden_dims'][2]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][2],
                        save_hidden=False,
                        token_len=args['model']['feature_extractor']['token_length'][2],
                        dim=args['model']['feature_extractor']['hidden_dims'][2],
                        depth=args['model']['feature_extractor']['depth'],
                        heads=args['model']['feature_extractor']['heads'],
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][2])
        )

        self.proj_v = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][1],
                      args['model']['feature_extractor']['hidden_dims'][1]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][1],
                        save_hidden=False,
                        token_len=args['model']['feature_extractor']['token_length'][1],
                        dim=args['model']['feature_extractor']['hidden_dims'][1],
                        depth=args['model']['feature_extractor']['depth'],
                        heads=args['model']['feature_extractor']['heads'],
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][1])
        )

        self.generate_proxy_modality = Generate_Proxy_Modality(args, args['model']['generate_proxy']['input_dim'],
                                                               args['model']['generate_proxy']['hidden_dim'],
                                                               args['model']['generate_proxy']['out_dim'])

        self.crossmodal_encoder = CrossmodalEncoder(proxy_dim=args['model']['crossmodal_encoder']['proxy_dim'],
                                                    text_dim=args['model']['crossmodal_encoder']['hidden_dims'][0],
                                                    audio_dim=args['model']['crossmodal_encoder']['hidden_dims'][2],
                                                    video_dim=args['model']['crossmodal_encoder']['hidden_dims'][1],
                                                    embed_dim=args['model']['crossmodal_encoder']['embed_dim'],
                                                    num_layers=args['model']['crossmodal_encoder']['num_layers'],
                                                    attn_dropout=args['model']['crossmodal_encoder']['attn_dropout'])

        self.fc1 = nn.Linear(args['model']['regression']['input_dim'], args['model']['regression']['hidden_dim'])
        self.fc2 = nn.Linear(args['model']['regression']['hidden_dim'], args['model']['regression']['out_dim'])
        self.dropout = nn.Dropout(args['model']['regression']['attn_dropout'])

    def forward(self, complete_input):
        vision, audio, language = complete_input
        h_l = self.proj_l(self.bertmodel(language))[:, :8]
        h_v = self.proj_v(vision)[:, :8]
        h_a = self.proj_a(audio)[:, :8]

        _, proxy_m, weight, proxy_c, *_ =self.generate_proxy_modality(
            h_l, h_v, h_a, h_l, h_v, h_a
        )
        feat = self.crossmodal_encoder(proxy_m, h_l, h_a, h_v, weight)
        return self.fc2(self.dropout(F.relu(self.fc1(torch.mean(feat, dim=1))))), proxy_c,weight