from flask import Flask, request, jsonify
import torch
from rl.agents.dqn import DQN
import os
import numpy as np


app = Flask('rlserve')
MODEL_PATH = os.environ.get('MODEL_PATH', '/models/dqn.pt')


# assume obs_dim=4 act_dim=2
model = DQN(4,2)
if os.path.exists(MODEL_PATH):
    state = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
else:
    model = None


@app.route('/infer', methods=['POST'])
def infer():
    if model is None:
        return jsonify({'action': 0, 'note': 'no model'})
    obs = np.array(request.json['obs'], dtype=np.float32)
    with torch.no_grad():
        t = torch.tensor(obs).unsqueeze(0)
        a = int(model(t).argmax(1).item())
    return jsonify({'action': a})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8501)