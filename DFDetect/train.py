import os
import logging
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import hydra
from omegaconf import DictConfig, OmegaConf

# Suppress harmless PyTorch flop_counter warning about missing Triton on Windows
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

from src.systems.dual_resnet34_logic import DualResNet34LightningSystem
from src.datamodules.deepfake_datamodule import DeepfakeDataModule

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Print the resolved config
    print("Running with Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # 1. Initialize Weights & Biases Logger
    wandb_logger = WandbLogger(
        project=cfg.project_name,
        name=cfg.run_name
    )

    # 2. Instantiate DataModule mapped to the configs
    data_module = DeepfakeDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers
    )
    
    data_module.setup()

    # 3. Instantiate Core Lightning System Logic loading Model configs
    model = DualResNet34LightningSystem(
        lr=cfg.model.lr,
        lambda1=cfg.model.lambda1,
        lambda2=cfg.model.lambda2,
        lambda3=cfg.model.lambda3,
        lambda5=cfg.model.lambda5,
        scheduler_patience=cfg.trainer.scheduler_patience
    )

    # 4. Setup Checkpointing using Trainer configs
    os.makedirs("logs/checkpoints", exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath="logs/checkpoints/",
        # Add run name to checkpoint filename for easy tracking
        filename=f"{cfg.run_name}-{{epoch:02d}}-{{val_loss:.4f}}",
        save_top_k=cfg.trainer.save_top_k,
        monitor="val_loss",
        mode="min",
        save_last=True # ALWAYS save a 'last.ckpt' at the end of every epoch to recover from crashes
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.trainer.patience,
        strict=False,
        verbose=True,
        mode="min"
    )

    # 5. Initialize the PyTorch Lightning Trainer mapped from config
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices
    )

    # 6. START TRAINING!
    print("Starting Training Pipeline...")
    
    last_ckpt_path = "logs/checkpoints/last.ckpt"
    if os.path.exists(last_ckpt_path):
        print(f"Found checkpoint at {last_ckpt_path}. Resuming training from where it left off!")
        trainer.fit(model, datamodule=data_module, ckpt_path=last_ckpt_path)
    else:
        trainer.fit(model, datamodule=data_module)

    # 7. Evaluate on the explicitly held-out Test Set
    print("\n--- Starting Test Phase Evaluation ---")
    trainer.test(model, datamodule=data_module, ckpt_path="best")

    # 8. Write to train_log.txt upon completion
    best_model_path = checkpoint_callback.best_model_path
    best_model_score = checkpoint_callback.best_model_score

    log_entry = f"--- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n"
    log_entry += f"Run Name    : {cfg.run_name}\n"
    log_entry += f"Best Model : {best_model_path}\n"
    log_entry += f"Best Val Loss : {best_model_score}\n"
    log_entry += "Parameters:\n"
    log_entry += OmegaConf.to_yaml(cfg)
    log_entry += "-" * 50 + "\n"

    print("Training finished. Writing summary to train_log.txt...")
    with open("train_log.txt", "a") as f:
        f.write(log_entry)

if __name__ == "__main__":
    main()