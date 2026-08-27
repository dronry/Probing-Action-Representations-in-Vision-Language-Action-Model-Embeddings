# Robot Learning Experiments — VLM Embedding Geometry for Action Generation

Research log investigating whether a frozen pi0/PaliGemma VLM's hidden-state
space contains usable structure for robot action prediction, and whether a
rectified-flow model can generate action-token embeddings in that space.

Originally run as a sequence of Kaggle/Colab notebook cells. Split here into
one folder per experiment so each stage of reasoning is easy to follow on
GitHub. Where a script depends on variables/state from an earlier script in
the same folder (i.e. it was never a standalone cell), that's noted in a
comment at the top of the file — see the folder's own README for run order.

## Folders, in the order the research happened

| Folder | Question |
|---|---|
| `00_setup/` | Environment setup (lerobot/pi0 track) |
| `exp1a_linear_probe/` | Do frozen VLM hidden states linearly predict ground-truth actions? |
| `exp1b_mlp_probe/` | Is the orientation gap in 1a a linearity problem or a representation problem? |
| `exp1c_orientation_ceiling/` | Why does orientation cap around R2~0.25-0.42 no matter the decoder? |
| `exp1d_vision_token_probe/` | Does orientation info live in vision tokens instead of the pooled language token? |
| `exp1_orientation_expert_design/` | Split-head architecture prototype: shared trunk + a dedicated orientation head |
| `exp1e_openvla_replication/` | Does the position/orientation split replicate on a different VLM backbone (OpenVLA)? |
| `exp2_encoder_geometry/` | Do newly-trained action-token embeddings land near real vocab tokens, or collapse? |
| `exp3_flow_trajectory_quality/` | Does a rectified flow through embedding space produce cleaner trajectories than raw action space? |

## Requirements

Run `00_setup/install.sh` first (lerobot/pi0 track — covers 1a-1d, the
orientation-expert design, exp2, exp3). `exp1e_openvla_replication/` needs a
separate, conflicting set of pins — see that folder's own setup script;
don't install both in the same environment.
