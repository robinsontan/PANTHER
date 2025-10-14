
import os

import torch
import torch.distributed as dist


class local_Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        file_dir,
        eval_file_dir=None,
        use_eval_user=False,
        eval_flag=False,
    ):
        super().__init__()

        self.file_dir = file_dir
        if eval_flag:
            self.filename_list = self.get_all_files(self.file_dir)
        else:
            self.filename_list = []
            if eval_file_dir is not None and use_eval_user:
                for r in range(len(os.listdir(eval_file_dir))):
                    source = f'{eval_file_dir}/rank_{r}'
                    for bid in range(len(os.listdir(source))):
                        self.filename_list.append(os.path.join(source, f"batch_{bid}.pt"))
            for r in range(len(os.listdir(file_dir))):
                source = f'{file_dir}/rank_{r}'
                for bid in range(len(os.listdir(source))):
                    self.filename_list.append(os.path.join(source, f"batch_{bid}.pt"))

    def __len__(self) -> int:
        return len(self.filename_list)

    def __getitem__(self, idx):
        data = self.filename_list[idx]
        # if dist.is_initialized():
        #     rank = dist.get_rank()
        #     map_location = f'cuda:{rank}'
        # else:
        #     map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        # map_location = 'cpu'
        # data = torch.load(self.filename_list[idx], map_location=map_location)
        # if data is None:
        #     import pdb; pdb.set_trace()
        return data
    def get_all_files(self, directory):
        file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                abs_file_path = os.path.abspath(file_path)
                file_paths.append(abs_file_path)
        return file_paths
