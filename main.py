"""
RiemannSD-Net: Speaker-Disentangled Metric Learning for Deepfake Source Verification.

Paper: "Disentangling Speaker Traits for Deepfake Source Verification
        via Chebyshev Polynomial and Riemannian Metric Learning"
Authors: Xi Xuan, Wenxin Zhang, Zhiyu Li, Jennifer Williams, Ville Hautamaki, Tomi H. Kinnunen

This is the main training entry point. It supports:
  - Multiple source-branch encoder architectures (ResNet34, ECAPA-TDNN, dual-branch variants)
  - Multiple loss functions including the proposed ChebySD-AAM and RiemannSD-AAM
  - Evaluation via cosine scoring with EER / minDCF metrics
"""

from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import warnings
import os

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.optim.lr_scheduler import StepLR

from module.feature import Mel_Spectrogram
from module.loader import SPK_datamodule
import score as score
from loss import softmax, amsoftmax
from loss.loss import *
from loss.SubcenterArcMarginProduct import SubcenterArcMarginProduct

warnings.filterwarnings("ignore")

# ─────────────────────────── Score logging path ───────────────────────────
file_path = '/xxuan-acoustics/interspeech2026/exps/exp85/score.txt'
folder_path = os.path.dirname(file_path)
os.makedirs(folder_path, exist_ok=True)
score_file = open(file_path, "a+")


class Task(LightningModule):
    """PyTorch Lightning module for source verification training and evaluation.

    The dual-branch models (dual_*) extract both a source embedding f_src
    (from a trainable encoder) and a speaker embedding f_spk (from a frozen
    speaker encoder), enabling the speaker-disentangled losses (ChebySD-AAM,
    RiemannSD-AAM, SPAAMsoftmax) to penalise speaker information leakage.
    """

    # Loss functions that require dual-branch (source + speaker) embeddings
    DUAL_LOSSES = [
        "SPAAMsoftmax", "ChebySPAAMSoftmax", "ChebySDAAMSoftmax",
        "RiemannianSPAAMSoftmax", "RiemannSDAAMSoftmax",
        "SP_amsoftmax", "RiemannianTangentAAM",
    ]

    def __init__(
        self,
        learning_rate: float = 0.2,
        weight_decay: float = 1.5e-6,
        batch_size: int = 32,
        num_workers: int = 10,
        max_epochs: int = 50,
        trial_path: str = "data/vox1_test.txt",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.trials = np.loadtxt(self.hparams.trial_path, str)
        self.mel_trans = Mel_Spectrogram()

        # ── Encoder selection ──
        self._build_encoder()

        # ── Loss function selection ──
        self._build_loss()

        self.epoch = 0

    def _build_encoder(self):
        """Instantiate the selected source encoder."""
        from module.ecapa_tdnn import ecapa_tdnn
        from module.dual_mamba import dual_mamba_cat
        from module.dual_ecapa import dual_ecapa_cat
        from module.dual_conformer import dual_conformer_cat
        from module.dual_transformer import dual_transformer_cat
        from module.dual_conv_conformer import dual_conv_conformer_cat
        from module.resnet34 import resnet34
        from module.dual_resnet34_cat import dual_resnet34_cat
        from module.dual_aasist_cat import dual_aasist_cat
        from module.dual_ReD_ecapa_cat import dual_ReD_ecapa_cat

        name = self.hparams.encoder_name
        dim = self.hparams.embedding_dim

        # Single-branch encoders
        if name == "resnet34":
            self.encoder = resnet34(embedding_dim=dim)
        elif name == "ecapa_tdnn":
            self.encoder = ecapa_tdnn(embedding_dim=dim)

        # Dual-branch encoders (source + frozen speaker branch)
        elif name == "dual_mamba_cat":
            self.encoder = dual_mamba_cat(
                embedding_dim=dim, num_blocks=self.hparams.num_blocks,
                input_layer=self.hparams.input_layer,
                pos_enc_layer_type=self.hparams.pos_enc_layer_type)
        elif name == "dual_ecapa_cat":
            self.encoder = dual_ecapa_cat(
                embedding_dim=dim, num_blocks=self.hparams.num_blocks,
                input_layer=self.hparams.input_layer,
                pos_enc_layer_type=self.hparams.pos_enc_layer_type)
        elif name == "dual_conformer_cat":
            self.encoder = dual_conformer_cat(
                embedding_dim=dim, num_blocks=self.hparams.num_blocks,
                input_layer=self.hparams.input_layer,
                pos_enc_layer_type=self.hparams.pos_enc_layer_type)
        elif name == "dual_transformer_cat":
            self.encoder = dual_transformer_cat(
                embedding_dim=dim, num_blocks=self.hparams.num_blocks,
                input_layer=self.hparams.input_layer,
                pos_enc_layer_type=self.hparams.pos_enc_layer_type)
        elif name == "dual_conv_conformer_cat":
            self.encoder = dual_conv_conformer_cat(
                embedding_dim=dim, num_blocks=self.hparams.num_blocks,
                input_layer=self.hparams.input_layer,
                pos_enc_layer_type=self.hparams.pos_enc_layer_type)
        elif name == "dual_resnet34_cat":
            self.encoder = dual_resnet34_cat(
                embedding_dim=dim, num_blocks=self.hparams.num_blocks,
                input_layer=self.hparams.input_layer,
                pos_enc_layer_type=self.hparams.pos_enc_layer_type)
        elif name == "dual_aasist_cat":
            self.encoder = dual_aasist_cat(
                embedding_dim=dim, num_blocks=self.hparams.num_blocks,
                input_layer=self.hparams.input_layer,
                pos_enc_layer_type=self.hparams.pos_enc_layer_type)
        elif name == "dual_ReD_ecapa_cat":
            self.encoder = dual_ReD_ecapa_cat(
                embedding_dim=dim, num_blocks=self.hparams.num_blocks,
                input_layer=self.hparams.input_layer,
                pos_enc_layer_type=self.hparams.pos_enc_layer_type)
        else:
            raise ValueError(f"Unknown encoder: {name}")

    def _build_loss(self):
        """Instantiate the selected loss function."""
        name = self.hparams.loss_name
        dim = self.hparams.embedding_dim
        n_cls = self.hparams.num_classes

        if name == "amsoftmax":
            self.loss_fun = amsoftmax(embedding_dim=dim, num_classes=n_cls)
        elif name == "SP_amsoftmax":
            self.loss_fun = SPAMSoftmax(
                embedding_dim=dim, num_classes=n_cls,
                margin=0.2, scale=30, lambda_val=0.5)
        elif name == "sc-aamsoftmax":
            self.loss_fun = SubcenterArcMarginProduct(
                in_features=dim, out_features=n_cls, K=2, s=30.0, m=0.20)
        elif name == "fuzzy-amsoftmax":
            self.loss_fun = FuzzyArcFaceLoss(
                in_features=dim, out_features=n_cls, s=30.0, m=0.20, tau=0.9)
        elif name == "AAMsoftmax":
            self.loss_fun = AAMsoftmax(n_class=n_cls, m=0.20, s=30.0)
        elif name == "SPAAMsoftmax":
            self.loss_fun = SPAAMsoftmax(
                in_feats=dim, n_class=n_cls, m=0.20, s=30.0, lambda_val=0.15)
        elif name == "ChebyAAMSoftmax":
            self.loss_fun = ChebyAAMSoftmax(
                in_feats=dim, n_class=n_cls, m=0.3, s=30.0)
        elif name in ("ChebySPAAMSoftmax", "ChebySDAAMSoftmax"):
            # ChebySD-AAM (proposed, Section 3.2)
            self.loss_fun = ChebySDAAMSoftmax(
                in_feats=dim, n_class=n_cls, m=0.3, s=30.0, lambda_val=0.15)
        elif name in ("RiemannianSPAAMSoftmax", "RiemannSDAAMSoftmax"):
            # RiemannSD-AAM (proposed, Section 3.3)
            self.loss_fun = RiemannSDAAMSoftmax(
                in_feats=dim, n_class=n_cls, spk_feats=192,
                m=0.35, s=30.0, lambda_val=0.2)
        elif name == "RiemannianTangentAAM":
            self.loss_fun = RiemannianTangentAAM(
                in_feats=dim, n_class=n_cls, spk_feats=192,
                m=0.35, s=30.0, lambda_orth=0.2)
        else:
            self.loss_fun = softmax(embedding_dim=dim, num_classes=n_cls)

    def forward(self, x):
        embedding = self.encoder(x, aug=True)
        return embedding

    def training_step(self, batch, batch_idx):
        waveform, waveform_for_ecapa, label = batch

        if self.hparams.loss_name in self.DUAL_LOSSES:
            # Dual-branch: source encoder produces (f_src, f_spk)
            source_embedding, speaker_embedding = self.encoder(
                waveform, waveform_for_ecapa, aug=True)
            loss, acc = self.loss_fun(source_embedding, speaker_embedding, label)
        else:
            # Single-branch: encoder produces embedding directly
            speaker_embedding = self.encoder(waveform, aug=True)
            loss, acc = self.loss_fun(speaker_embedding, label)

        self.log('train_loss', loss, prog_bar=True)
        self.log('acc', acc, prog_bar=True)
        return loss

    def on_test_epoch_start(self):
        return self.on_validation_epoch_start()

    def on_validation_epoch_start(self):
        self.index_mapping = {}
        self.eval_vectors = []

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        x, path = batch
        path = path[0]
        with torch.no_grad():
            self.encoder.eval()
            x = self.encoder(x, aug=False)
        x = x.detach().cpu().numpy()[0]
        self.eval_vectors.append(x)
        self.index_mapping[path] = batch_idx

    def validation_epoch_end(self, outputs):
        num_gpus = 1
        eval_vectors = [None for _ in range(num_gpus)]
        dist.all_gather_object(eval_vectors, self.eval_vectors)
        eval_vectors = np.vstack(eval_vectors)

        table = [None for _ in range(num_gpus)]
        dist.all_gather_object(table, self.index_mapping)

        index_mapping = {}
        for i in table:
            index_mapping.update(i)

        # Mean-center the embeddings
        eval_vectors = eval_vectors - np.mean(eval_vectors, axis=0)

        labels, scores = score.cosine_score(
            self.trials, index_mapping, eval_vectors)

        EER, threshold = score.compute_eer(labels, scores)
        print("===Test experimental results=======")
        print("\nEER= {:.6f}% ".format(EER * 100))
        self.log("cosine_eer", EER * 100)

        minDCF1, threshold1 = score.compute_minDCF(labels, scores, p_target=0.1)
        print("minDCF(0.1): {:.6f} with threshold {:.4f}".format(minDCF1, threshold1))
        self.log("cosine_minDCF(0.1)", minDCF1)

        minDCF2, threshold2 = score.compute_minDCF(labels, scores, p_target=0.5)
        print("minDCF(0.5)= {:.6f} ".format(minDCF2))
        self.log("cosine_minDCF(0.5)", minDCF2)

        minDCF3, threshold3 = score.compute_minDCF(labels, scores, p_target=0.01)
        print("minDCF(0.01): {:.6f} with threshold {:.4f}".format(minDCF3, threshold3))
        self.log("cosine_minDCF(0.01)", minDCF3)

        minDCF4, threshold4 = score.compute_minDCF(labels, scores, p_target=0.001)
        print("minDCF(0.001): {:.6f} with threshold {:.4f}".format(minDCF4, threshold4))
        self.log("cosine_minDCF(0.001)", minDCF4)
        print("====================================")

        score_file.write("===============Epoch {%d}================\n" % self.epoch)
        score_file.write("cosine EER: {:.4f}% with threshold {:.6f}\n".format(EER * 100, threshold))
        score_file.write("cosine minDCF(0.1): {:.4f} with threshold {:.6f}\n".format(minDCF1, threshold1))
        score_file.write("cosine minDCF(0.5): {:.4f} with threshold {:.6f}\n".format(minDCF2, threshold2))
        score_file.write("cosine minDCF(0.01): {:.4f} with threshold {:.6f}\n".format(minDCF3, threshold3))
        score_file.write("cosine minDCF(0.001): {:.4f} with threshold {:.6f}\n".format(minDCF4, threshold4))
        score_file.flush()

        self.epoch += 1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = StepLR(optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # Warm up learning rate
        if self.trainer.global_step < self.hparams.warmup_step:
            lr_scale = min(1., float(self.trainer.global_step + 1) / float(self.hparams.warmup_step))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.learning_rate
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        (args, _) = parser.parse_known_args()

        # Model architecture
        parser.add_argument("--embedding_dim", default=512, type=int)
        parser.add_argument("--num_classes", type=int, default=41)
        parser.add_argument("--num_blocks", type=int, default=6)
        parser.add_argument("--input_layer", type=str, default="conv2d2")
        parser.add_argument("--pos_enc_layer_type", type=str, default="rel_pos")

        # Training
        parser.add_argument("--num_workers", default=28, type=int)
        parser.add_argument("--second", type=int, default=3)
        parser.add_argument('--step_size', type=int, default=1)
        parser.add_argument('--gamma', type=float, default=0.9)
        parser.add_argument("--batch_size", type=int, default=200)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--warmup_step", type=float, default=2000)
        parser.add_argument("--weight_decay", type=float, default=0.000001)

        # Paths
        parser.add_argument("--save_dir", type=str,
                            default="/xxuan-acoustics/interspeech2026/exps/exp85")
        parser.add_argument("--checkpoint_path", type=str, default=None)

        # Loss and encoder selection
        parser.add_argument("--loss_name", type=str, default="ChebySDAAMSoftmax",
                            help="Loss function: amsoftmax | AAMsoftmax | sc-aamsoftmax | "
                                 "SPAAMsoftmax | ChebyAAMSoftmax | ChebySDAAMSoftmax | "
                                 "RiemannSDAAMSoftmax | RiemannianTangentAAM")
        parser.add_argument("--encoder_name", type=str, default="resnet34",
                            help="Encoder: resnet34 | ecapa_tdnn | dual_mamba_cat | "
                                 "dual_ecapa_cat | dual_conformer_cat | dual_transformer_cat | "
                                 "dual_conv_conformer_cat | dual_resnet34_cat | "
                                 "dual_aasist_cat | dual_ReD_ecapa_cat")

        # Data
        parser.add_argument("--train_csv_path", type=str,
                            default="/xxuan-acoustics/dataset/MLAAD_v8/ptotocal/train_protocol_mapped_all.csv")
        parser.add_argument("--trial_path", type=str,
                            default="/xxuan-acoustics/dataset/MLAAD_v8/ptotocal/pairs_protocal/seen_test.txt")
        parser.add_argument("--score_save_path", type=str, default=None)

        # Mode
        parser.add_argument('--eval', action='store_true', default=False)
        parser.add_argument('--aug', default="True")
        return parser


def cli_main():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = Task.add_model_specific_args(parser)
    args = parser.parse_args()

    model = Task(**args.__dict__)

    if args.checkpoint_path is not None:
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict, strict=True)
        print("Loaded weights from {}".format(args.checkpoint_path))

    assert args.save_dir is not None

    checkpoint_callback = ModelCheckpoint(
        monitor='cosine_eer',
        save_top_k=5,
        mode='min',
        filename="{epoch}_{cosine_eer:.4f}",
        dirpath=args.save_dir,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    print("Data augmentation: {}".format(args.aug))
    dm = SPK_datamodule(
        train_csv_path=args.train_csv_path,
        trial_path=args.trial_path,
        second=args.second,
        aug=args.aug,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pairs=False,
    )

    AVAIL_GPUS = torch.cuda.device_count()
    print("Available GPUs:", AVAIL_GPUS)

    trainer = Trainer(
        max_epochs=100,
        strategy=DDPStrategy(find_unused_parameters=True),
        accelerator="gpu",
        devices=[0],
        num_sanity_val_steps=0,
        sync_batchnorm=True,
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=args.save_dir,
        accumulate_grad_batches=1,
        log_every_n_steps=25,
    )

    if args.eval:
        trainer.test(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    cli_main()
