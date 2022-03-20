from sol.start_of_line_finder import StartOfLineFinder
from lf.line_follower import LineFollower
from hw import cnn_lstm

from utils import safe_load

import os


def init_model(model, only_load=None):
    base_0 = 16
    base_1 = 16

    sol = None
    lf = None
    hw = None

    if only_load is None or only_load == 'sol' or 'sol' in only_load:
        sol = StartOfLineFinder(base_0, base_1)
        # sol_state = safe_load.torch_state(os.path.join(config['training']['snapshot'][sol_dir], "sol.pt"))
        sol_state = safe_load.torch_state(model.sol.file)

        sol.load_state_dict(sol_state)
        sol.cuda()

    if only_load is None or only_load == 'lf' or 'lf' in only_load:
        lf = LineFollower(60)
        # lf_state = safe_load.torch_state(os.path.join(config['training']['snapshot'][lf_dir], "lf.pt"))
        lf_state = safe_load.torch_state(model.lf.file)
        model_dict_clone = lf_state['cnn'].copy()
        for key, value in model_dict_clone.items():
            if key.endswith(('running_mean', 'running_var')):
                del lf_state['cnn'][key]

        if 'cnn' in lf_state:
            new_state = {}
            for k, v in lf_state.items():
                if k == 'cnn':
                    for k2, v2 in v.items():
                        new_state[k + "." + k2] = v2
                if k == 'position_linear':
                    for k2, v2 in v.state_dict().items():
                        new_state[k + "." + k2] = v2

            lf_state = new_state

        lf.load_state_dict(lf_state)
        lf.cuda()

    if only_load is None or only_load == 'hw' or 'hw' in only_load:
        config = {
            'cnn_out_size': 1024,
            'num_of_channels': 3,
            'num_of_outputs': 197
        }
        hw = cnn_lstm.create_model(config)
        hw_state = safe_load.torch_state(model.hw.file)
        hw.load_state_dict(hw_state)
        hw.cuda()

    return sol, lf, hw
