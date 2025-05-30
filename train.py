import yaml
import logging
from ultralytics import YOLO
import argparse
import os

# --------------------------
# Setup Logging
# --------------------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "train.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --------------------------
# Load Training Config
# --------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    logger.info("Loaded training config from config.yaml")

# --------------------------
# Train Function
# --------------------------
def train(resume=False, checkpoint_path=None):
    try:
        if checkpoint_path:
            if os.path.exists(checkpoint_path):
                logger.info(f"Resuming from checkpoint: {checkpoint_path}")
                model = YOLO(checkpoint_path)
            else:
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        else:
            logger.info(f"Starting training with model: {config['model']}")
            model = YOLO(config["model"])

        model.train(
            data=config["data"],
            imgsz=config["imgsz"],
            epochs=config["epochs"],
            batch=config["batch"],
            project=config["project"],
            name=config["name"],
            resume=resume
        )
        logger.info("Training completed.")

    except Exception as e:
        logger.exception(f"Training failed: {e}")

# --------------------------
# Entry Point
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to a specific checkpoint (e.g., last.pt)")
    args = parser.parse_args()

    logger.info("Training script started.")
    train(resume=args.resume, checkpoint_path=args.checkpoint_path)


# Run this command to train the model:
# python train.py --resume    
# python train.py --resume --checkpoint_path runs/train/amd_yolov8m/weights/last.pt

