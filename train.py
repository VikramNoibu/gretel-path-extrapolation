#!/usr/bin/env python
# coding: utf-8
# %%
from main import *
import random
import string

# ## Setup Config
# %%
domain = 'pollini'
config_path = "config/" + domain
config = Config()
config.load_from_file(config_path)

logging = SummaryWriter()

# Hyperparameters
config.lr = 0.01
config.batch_size = 24
config.self_loop_weight = 0.01
config.target_prediction = "target"
config.loss = 'target_only'
config.number_observations = 5

random_name = ''.join(random.choice(string.ascii_lowercase) for i in range(6))

# ## Setup Model

# %%
TRAJ_MAX_LEN = 99
graph, trajectories, pairwise_node_features, _ = load_data(config)

model = create_model(graph, pairwise_node_features, config)
model = model.to(config.device)
train_trajectories, valid_trajectories, test_trajectories = trajectories
use_validation_set = len(valid_trajectories) > 0

graph = graph.to(config.device)
given_as_target, siblings_nodes = None, None

if config.enable_checkpointing:
    chkpt_dir = os.path.join(
        config.workspace, config.checkpoint_directory, config.name)
    os.makedirs(chkpt_dir, exist_ok=True)

optimizer = create_optimizer(model.parameters(), config)

if config.restore_from_checkpoint:
    filename = input("Checkpoint file: ")
    checkpoint_data = torch.load(filename)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
    print("Loaded parameters from checkpoint")


def create_evaluator():
    return Evaluator(
        graph.n_node,
        given_as_target=given_as_target,
        siblings_nodes=siblings_nodes,
        config=config,
    )


if config.compute_baseline:
    display_baseline(config, graph, train_trajectories,
                     test_trajectories, create_evaluator())

graph = graph.add_self_loops(
    degree_zero_only=config.self_loop_deadend_only, edge_value=config.self_loop_weight)
evaluate(model, graph, test_trajectories, pairwise_node_features,
         create_evaluator, dataset="TEST")

# ## Train
# %%
for epoch in range(config.number_epoch):
    model.train()

    print_cum_loss = 0.0
    print_num_preds = 0
    print_time = time.time()
    print_every = len(
        train_trajectories) // config.batch_size // config.print_per_epoch

    trajectories_shuffle_indices = np.arange(len(train_trajectories))
    if config.shuffle_samples:
        np.random.shuffle(trajectories_shuffle_indices)

    for iteration, batch_start in enumerate(range(0, len(trajectories_shuffle_indices) -
                                                  config.batch_size + 1, config.batch_size)):

        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=config.device)

        for i in range(batch_start, batch_start + config.batch_size):
            trajectory_idx = trajectories_shuffle_indices[i]
            observations = train_trajectories[trajectory_idx]
            length = train_trajectories.lengths[trajectory_idx]

            number_steps = None
            if config.rw_edge_weight_see_number_step or config.rw_expected_steps:
                if config.use_shortest_path_distance:
                    number_steps = (
                        train_trajectories.leg_shortest_lengths(
                            trajectory_idx).float() * 1.1
                    ).long()
                else:
                    number_steps = train_trajectories.leg_lengths(
                        trajectory_idx)

            observed, starts, targets = generate_masks(
                trajectory_length=observations.shape[0],
                number_observations=config.number_observations,
                predict=config.target_prediction,
                with_interpolation=config.with_interpolation,
                device=config.device,
            )

            diffusion_graph = graph if not config.diffusion_self_loops else graph.add_self_loops()

            predictions, potentials, rw_weights = model(
                observations,
                graph,
                diffusion_graph,
                observed=observed,
                starts=starts,
                targets=targets,
                pairwise_node_features=pairwise_node_features,
                number_steps=number_steps,
            )

            print_num_preds += starts.shape[0]

            l = (compute_loss(config.loss, train_trajectories, observations, predictions, starts, targets, rw_weights, trajectory_idx,)
                 / starts.shape[0])
            loss += l

        loss /= config.batch_size
        print_cum_loss += loss.item()
        loss.backward()
        optimizer.step()

        if (iteration + 1) % print_every == 0:
            print_loss = print_cum_loss / print_every
            print_loss /= print_num_preds
            pred_per_second = 1.0 * print_num_preds / \
                (time.time() - print_time)

            print_cum_loss = 0.0
            print_num_preds = 0
            print_time = time.time()

            progress_percent = int(
                100.0 * ((iteration + 1) // print_every) /
                config.print_per_epoch
            )

            print(
                f"Progress {progress_percent}% | iter {iteration} | {pred_per_second:.1f} pred/s | loss {config.loss} {print_loss}"
            )
    # VALID and TEST metrics computation
    test_evaluator, node_acc_dist = evaluate(
        model, graph, test_trajectories, pairwise_node_features, create_evaluator, dataset="TEST",)
    # Checkpointing
    if config.enable_checkpointing and epoch % config.chechpoint_every_num_epoch == 0:
        print("Checkpointing...")
        directory = os.path.join(
            config.workspace, config.checkpoint_directory, config.name)
        chkpt_file = os.path.join(directory, f"{epoch:04d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            chkpt_file,
        )
        config_file = os.path.join(directory, "config")
        config.save_to_file(config_file)

        metrics_file = os.path.join(directory, f"{epoch:04d}.txt")
        with open(metrics_file, "w") as f:
            f.write(test_evaluator.to_string())
        print(colored(f"Checkpoint saved in {chkpt_file}", "blue"))
    # log hyperparameters
    logging.add_hparams({'batch_size': config.batch_size, 'lr': config.lr,
                        'self_loop_weight': config.self_loop_weight,
                         'loss': config.loss,
                         'target': config.target_prediction, "n_obs": config.number_observations},
                        {"loss": loss}, name=random_name)
logging.close()

# %%
temp = torch.zeros(10)
