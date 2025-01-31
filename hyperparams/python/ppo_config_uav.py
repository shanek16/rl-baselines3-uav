import torch

hyperparams = {
    "Rand_cycle-v0": dict(
    normalize= "{'norm_obs': True, 'norm_reward': False, 'norm_obs_keys' : ['uav1_state', 'uav2_state', 'target1_position', 'target2_position', 'target3_position', 'battery']}",
    n_envs= 16,
    n_timesteps= 1e6,
    policy= "MultiInputPolicy",
    n_steps= 1024,
    batch_size= 64,
    gae_lambda= 0.98,
    gamma= 0.999,
    n_epochs= 4,
    ent_coef= 0.01,
    )
}