from typing import Optional, Dict
import os

class TopKCheckpointManager:
    def __init__(self,
            save_dir,
            monitor_key: str,
            mode='min',
            k=1,
            format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt'
        ):
        assert mode in ['max', 'min']
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        # self.path_value_map = dict()
        self.path_value_map_arm1 = dict()
        self.path_value_map_arm2 = dict()
    
    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None

        value = data[self.monitor_key]
        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**data))
        
        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_path] = value
            return ckpt_path
        
        # at capacity
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == 'max':
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None
        else:
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            if os.path.exists(delete_path):
                os.remove(delete_path)
            return ckpt_path

    def get_ckpt_paths(self, data: Dict[str, float]) -> (Optional[str], Optional[str]):
        """
        根据 data 中的 "train_loss_arm1" 与 "train_loss_arm2" 返回两个 checkpoint 路径，
        文件名中分别带有 "arm1" 和 "arm2" 前缀。
        """
        if self.k == 0:
            return None, None

        # 获取各 arm 的监控值
        value_arm1 = data.get("train_loss_arm1", None)
        value_arm2 = data.get("train_loss_arm2", None)
        if value_arm1 is None or value_arm2 is None:
            return None, None

        # 格式化文件名时，将 "train_loss" 替换为各自的值，并添加 arm 标识前缀
        ckpt_path_arm1 = os.path.join(
            self.save_dir,
            "arm1_" + self.format_str.format(**{**data, "train_loss": data["train_loss_arm1"]})
        )
        ckpt_path_arm2 = os.path.join(
            self.save_dir,
            "arm2_" + self.format_str.format(**{**data, "train_loss": data["train_loss_arm2"]})
        )

        # 处理 arm1：判断是否需要更新路径
        if len(self.path_value_map_arm1) < self.k:
            self.path_value_map_arm1[ckpt_path_arm1] = value_arm1
            arm1_ret = ckpt_path_arm1
        else:
            sorted_map_arm1 = sorted(self.path_value_map_arm1.items(), key=lambda x: x[1])
            delete_path_arm1 = None
            if self.mode == 'max':
                if value_arm1 > sorted_map_arm1[0][1]:
                    delete_path_arm1 = sorted_map_arm1[0][0]
            else:  # mode == 'min'
                if value_arm1 < sorted_map_arm1[-1][1]:
                    delete_path_arm1 = sorted_map_arm1[-1][0]
            if delete_path_arm1 is None:
                arm1_ret = None
            else:
                del self.path_value_map_arm1[delete_path_arm1]
                self.path_value_map_arm1[ckpt_path_arm1] = value_arm1
                if not os.path.exists(self.save_dir):
                    os.mkdir(self.save_dir)
                if os.path.exists(delete_path_arm1):
                    os.remove(delete_path_arm1)
                arm1_ret = ckpt_path_arm1

        # 处理 arm2：类似处理
        if len(self.path_value_map_arm2) < self.k:
            self.path_value_map_arm2[ckpt_path_arm2] = value_arm2
            arm2_ret = ckpt_path_arm2
        else:
            sorted_map_arm2 = sorted(self.path_value_map_arm2.items(), key=lambda x: x[1])
            delete_path_arm2 = None
            if self.mode == 'max':
                if value_arm2 > sorted_map_arm2[0][1]:
                    delete_path_arm2 = sorted_map_arm2[0][0]
            else:  # mode == 'min'
                if value_arm2 < sorted_map_arm2[-1][1]:
                    delete_path_arm2 = sorted_map_arm2[-1][0]
            if delete_path_arm2 is None:
                arm2_ret = None
            else:
                del self.path_value_map_arm2[delete_path_arm2]
                self.path_value_map_arm2[ckpt_path_arm2] = value_arm2
                if not os.path.exists(self.save_dir):
                    os.mkdir(self.save_dir)
                if os.path.exists(delete_path_arm2):
                    os.remove(delete_path_arm2)
                arm2_ret = ckpt_path_arm2

        return arm1_ret, arm2_ret