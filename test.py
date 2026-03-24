"""
RiemannSD-Net: Evaluation script with four disentanglement protocols.

Evaluates a trained model on four source verification protocols:
  P-I:   Seen sources, same speaker
  P-II:  Seen sources, different speakers
  P-III: Unseen sources, same speaker
  P-IV:  Unseen sources, different speakers

Reports EER, AUC, ACC, F1, and minDCF for each protocol.
"""

from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import warnings
import os

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
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
file_path = '/xxuan-acoustics/interspeech2026/exps/exp76/all_score.txt'
folder_path = os.path.dirname(file_path)
os.makedirs(folder_path, exist_ok=True)
score_file = open(file_path, "a+")

# ─────────────────────────── Protocol paths ───────────────────────────
PROTOCOL_BASE = "/xxuan-acoustics/dataset/MLAAD_v8/ptotocal/pairs_protocal/large_version/Source-Speaker_evaluation_protocal/Final"
PROTOCOL_FILES = {
    "P-I (Seen-Same)":     f"{PROTOCOL_BASE}/seen_seen_same_speaker.txt",
    "P-II (Seen-Diff)":    f"{PROTOCOL_BASE}/seen_seen_diff_speaker.txt",
    "P-III (Unseen-Same)": f"{PROTOCOL_BASE}/unseen_unseen_same_speaker.txt",
    "P-IV (Unseen-Diff)":  f"{PROTOCOL_BASE}/unseen_unseen_diff_speaker.txt",
}


class Task(LightningModule):
    """Evaluation task with four disentanglement protocols."""

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

        # Load four protocol trial lists
        self.protocol_trials = {}
        for name, path in PROTOCOL_FILES.items():
            self.protocol_trials[name] = np.loadtxt(path, str)

        self.mel_trans = Mel_Spectrogram()
        self._build_encoder()
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

        if name == "resnet34":
            self.encoder = resnet34(embedding_dim=dim)
        elif name == "ecapa_tdnn":
            self.encoder = ecapa_tdnn(embedding_dim=dim)
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
        elif name == "sc-aamsoftmax":
            self.loss_fun = SubcenterArcMarginProduct(
                in_features=dim, out_features=n_cls, K=2, s=30.0, m=0.20)
        elif name == "AAMsoftmax":
            self.loss_fun = AAMsoftmax(n_class=n_cls, m=0.20, s=30.0)
        elif name == "ChebyAAMSoftmax":
            self.loss_fun = ChebyAAMSoftmax(
                in_feats=dim, n_class=n_cls, m=0.3, s=30.0)
        elif name in ("ChebySDAAMSoftmax"):
            self.loss_fun = ChebySDAAMSoftmax(
                in_feats=dim, n_class=n_cls, m=0.3, s=30.0, lambda_val=0.15)
        elif name in ("RiemannSDAAMSoftmax"):
            self.loss_fun = RiemannSDAAMSoftmax(
                in_feats=dim, n_class=n_cls, spk_feats=192,
                m=0.35, s=30.0, lambda_val=0.2)

        else:
            self.loss_fun = softmax(embedding_dim=dim, num_classes=n_cls)

    def forward(self, x):
        embedding = self.encoder(x, aug=True)
        return embedding

    def training_step(self, batch, batch_idx):
        waveform, waveform_for_ecapa, label = batch

        if self.hparams.loss_name in self.DUAL_LOSSES:
            source_embedding, speaker_embedding = self.encoder(
                waveform, waveform_for_ecapa, aug=True)
            loss, acc = self.loss_fun(source_embedding, speaker_embedding, label)
        else:
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
            source_embedding, _ = self.encoder(x, waveform_for_ecapa=None, aug=False)
        x = source_embedding.detach().cpu().numpy()[0]
        self.eval_vectors.append(x)
        self.index_mapping[path] = batch_idx

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def validation_epoch_end(self, outputs):
        print("validation_epoch_end running...")
        num_gpus = 1

        eval_vectors = [None for _ in range(num_gpus)]
        dist.all_gather_object(eval_vectors, self.eval_vectors)
        eval_vectors = np.vstack(eval_vectors)

        table = [None for _ in range(num_gpus)]
        dist.all_gather_object(table, self.index_mapping)

        index_mapping = {}
        for i in table:
            index_mapping.update(i)

        # Mean-center embeddings
        eval_vectors = eval_vectors - np.mean(eval_vectors, axis=0)

        def compute_extra_metrics(labels, scores, threshold):
            try:
                auc = roc_auc_score(labels, scores)
            except ValueError:
                auc = 0.0
            predictions = (scores >= threshold).astype(int)
            acc = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions)
            return acc, f1, auc

        # Evaluate all four protocols
        for protocol_name, trials in self.protocol_trials.items():
            print(f"\n{'='*30} {protocol_name} {'='*30}")
            labels, scores = score.cosine_score(trials, index_mapping, eval_vectors)
            EER, threshold = score.compute_eer(labels, scores)
            acc, f1, auc = compute_extra_metrics(labels, scores, threshold)

            print(f"EER= {EER * 100:.6f}%")
            print(f"AUC: {auc:.6f}")
            print(f"ACC: {acc * 100:.6f}%")
            print(f"F1: {f1:.6f}")

            minDCF1, th1 = score.compute_minDCF(labels, scores, p_target=0.1)
            minDCF2, th2 = score.compute_minDCF(labels, scores, p_target=0.5)
            minDCF3, th3 = score.compute_minDCF(labels, scores, p_target=0.01)
            minDCF4, th4 = score.compute_minDCF(labels, scores, p_target=0.001)

            print(f"minDCF(0.1): {minDCF1:.6f} threshold {th1:.4f}")
            print(f"minDCF(0.5): {minDCF2:.6f} threshold {th2:.4f}")
            print(f"minDCF(0.01): {minDCF3:.6f} threshold {th3:.4f}")
            print(f"minDCF(0.001): {minDCF4:.6f} threshold {th4:.4f}")

            score_file.write(f"==============={protocol_name}================\n")
            score_file.write(f"cosine EER: {EER * 100:.4f}% threshold {threshold:.6f}\n")
            score_file.write(f"cosine AUC: {auc:.6f}\n")
            score_file.write(f"cosine ACC: {acc * 100:.4f}%\n")
            score_file.write(f"cosine F1: {f1:.6f}\n")
            score_file.write(f"cosine minDCF(0.1): {minDCF1:.4f} threshold {th1:.6f}\n")
            score_file.write(f"cosine minDCF(0.5): {minDCF2:.4f} threshold {th2:.6f}\n")
            score_file.write(f"cosine minDCF(0.01): {minDCF3:.4f} threshold {th3:.6f}\n")
            score_file.write(f"cosine minDCF(0.001): {minDCF4:.4f} threshold {th4:.6f}\n")
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
                            default="/xxuan-acoustics/interspeech2026/exps/exp87")
        parser.add_argument("--checkpoint_path", type=str, default=None)

        # Loss and encoder
        parser.add_argument("--loss_name", type=str, default="ChebySDAAMSoftmax")
        parser.add_argument("--encoder_name", type=str, default="dual_ReD_ecapa_cat")

        # Data
        parser.add_argument("--train_csv_path", type=str,
                            default="/xxuan-acoustics/dataset/MLAAD_v8/ptotocal/train_protocol_mapped_all.csv")
        parser.add_argument("--trial_path", type=str,
                            default="/xxuan-acoustics/dataset/MLAAD_v8/ptotocal/pairs_protocal/seen_test.txt")
        parser.add_argument("--score_save_path", type=str, default=None)

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

    trainer.validate(model, datamodule=dm)


if __name__ == "__main__":
    cli_main()
