from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "models" / "yolov8n.pt"
LOG_PATH = ROOT_DIR / "logs" / "app.log"