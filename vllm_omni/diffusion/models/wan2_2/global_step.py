# global_step.py
global_step = -1
current_noise_stage = "high"
# Post-patch latent shape (N_t, N_h, N_w) for 3D block sparse attention.
# Set by WanTransformer3DModel.forward(), read by FlashAttentionImpl.
latent_shape = None