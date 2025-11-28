import os
import torch
from pathlib import Path
from rl.agents.dqn import DQN

# Automatically find project root and model path
ROOT_DIR = Path(__file__).resolve().parents[2]  # D:\Ayush_AIML\Rl_BASED_TRAFIC_OPTIMIZATION_SYSTEM
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "dqn.pt"

MODEL_PATH = os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH))


def load_model(device="cpu"):
    """
    Attempt to load model weights from MODEL_PATH.
    Returns (model_instance, model_loaded_bool)
    """
    print(f"🔍 Looking for model at: {MODEL_PATH}")
    try:
        obs_dim = int(os.environ.get("OBS_DIM", 4))
        act_dim = int(os.environ.get("ACT_DIM", 2))
        model = DQN(obs_dim, act_dim).to(device)

        path = Path(MODEL_PATH)
        if not path.exists():
            print(f"⚠️ Model path not found: {path}")
            return model, False

        # Load the model
        state = torch.load(path, map_location=device)
        if isinstance(state, dict):
            model.load_state_dict(state)
            print(f"✅ Loaded model state_dict from {path}")
            return model, True
        else:
            print(f"✅ Loaded full model object from {path}")
            return state, True

    except Exception as e:
        print(f"❌ Model load failed: {e}")
        # Return a default model (untrained) to keep the backend working
        return DQN(4, 2), False
