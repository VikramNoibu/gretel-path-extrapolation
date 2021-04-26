# %%
from main import *
# %%
k = 5
min_len = 10
model = read_model()
trajs = get_trajs()  # read the error trajs

# %%
for traj in trajs:
    obs = obs[:min_len]
    actual = obs[min_len:min_len+k]
    for i in range(k):
        preds = model(obs)
        obs = obs[1:] + preds
        ksteps.append(preds)

    acc = [int(kstep == act) for kstep, act in zip(ksteps, actual)]
