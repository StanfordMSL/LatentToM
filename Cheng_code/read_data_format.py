# import zarr
# import numpy as np
#
# zarr_path = "/home/cheng/Chengyang/diffusion_policy/diffusion_policy_vanilla/data/pusht_real/real_pusht_20230105/replay_buffer.zarr/data"
# replay_buffer = zarr.open(zarr_path, mode='r')
#
# replay_dict = {key: np.array(replay_buffer[key]) for key in replay_buffer.keys()}
#
# for key, value in replay_dict.items():
#     print(f"{key}: shape={value.shape}, dtype={value.dtype}")
#
# print(replay_dict["action"].shape)
# print(replay_dict["robot_eef_pose"].shape)
#
# print(replay_dict["action"][8])
# print(replay_dict["robot_eef_pose"][10])


##########################################################
#             check the zarr file tree and components
##########################################################
import zarr

# Load replay_buffer.zarr
dataset_path = "/home/admin/Chengyang/diffusion_policy/diffusion_policy_pickcup/data/xarm_multi/xarm_push_particle/replay_buffer.zarr"
replay_buffer = zarr.open(dataset_path, mode='r')

# Print available datasets
print(replay_buffer.tree())

# # Print timestamps
# timestamps = replay_buffer['data/timestamp'][:]
# print(timestamps[:10])  # Show first 10 timestamps

# List all components inside replay_buffer.zarr
for key in replay_buffer['data']:
    data = replay_buffer[f'data/{key}']
    print(f"Component: {key}")
    print(f"  Shape: {data.shape}")
    print(f"  Data Type: {data.dtype}")
    print(f"  First 5 values:\n{data[:5]}")
    print("-" * 40)

# check what is in meta file
if 'meta' in replay_buffer:
    for key in replay_buffer['meta']:
        metadata = replay_buffer[f'meta/{key}'][:]
        print(f"Metadata: {key}")
        print(f"Metadata: {metadata}")
        # print(metadata)

# check chunk size
print("-" * 40)
print(replay_buffer['data/action'].chunks)
#

#
# import os
# import zarr
# from diffusion_policy.common.replay_buffer import ReplayBuffer
#
# dataset_path = "/home/admin/Chengyang/diffusion_policy/diffusion_policy_pickcup/data/xarm_real/xarm_pickcup_20250224/replay_buffer.zarr"
# assert os.path.exists(dataset_path), f"❌ Dataset path {dataset_path} does not exist!"
#
# in_replay_buffer = zarr.open(dataset_path, mode='r')
# print("✅ Replay buffer loaded!")
# print(in_replay_buffer.tree())  # Ensure the dataset is properly structured
#
# out_store = zarr.MemoryStore()
# out_replay_buffer = ReplayBuffer.copy_from_store(
#     src_store=in_replay_buffer.store,
#     store=out_store,
#     keys=['action', 'robot_eef_pos', 'robot_eef_quat', 'robot_gripper_qpos', 'timestamp'],  # Ensure these keys exist
# )



