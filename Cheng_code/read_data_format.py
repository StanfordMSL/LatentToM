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
dataset_path = "/home/cheng/Chengyang/diffusion_policy/diffusion_policy_simplePushT/data/xarm_real/xarm_pusht_20250212/replay_buffer.zarr"
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


# import os
#
# dataset_path = "/home/cheng/Chengyang/diffusion_policy/diffusion_policy_simplePushT/data/xarm_real/xarm_pusht_20250211"
# expected_dirs = ["replay_buffer.zarr", "videos"]
# expected_files = ["replay_buffer.zarr/data/action", "replay_buffer.zarr/data/robot_eef_pose", "replay_buffer.zarr/data/timestamp"]
#
# # Check if dataset path exists
# if not os.path.exists(dataset_path):
#     print(f"❌ Dataset path {dataset_path} is missing!")
# else:
#     print(f"✅ Dataset path {dataset_path} exists.")
#
# # Check required directories
# for d in expected_dirs:
#     full_path = os.path.join(dataset_path, d)
#     if not os.path.exists(full_path):
#         print(f"❌ Missing directory: {full_path}")
#     else:
#         print(f"✅ Found: {full_path}")
#
# # Check required files
# for f in expected_files:
#     full_path = os.path.join(dataset_path, f)
#     if not os.path.exists(full_path):
#         print(f"❌ Missing file: {full_path}")
#     else:
#         print(f"✅ Found: {full_path}")


