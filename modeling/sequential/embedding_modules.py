

import abc
import hashlib
import os

# from torchrec.modules.crossnet import LowRankCrossNet
from collections import OrderedDict
from typing import Dict, List, Optional, Union

import torch
from sklearn.preprocessing import StandardScaler

from modeling.initialization import truncated_normal
from trainer.tokenizer import *


class SENet(torch.nn.Module):
    def __init__(
        self, 
        num_fields: int,
        reduction_ratio: int = 2,
        # reduced_size: Optional[int] = None,
        reduced_size = None,
    ):
        super().__init__()
        self.reduced_size = reduced_size if reduced_size else max(1, num_fields // reduction_ratio)
        self.num_fields = num_fields  # K
        self.excitation = torch.nn.Sequential(  # [B,N,1,K] -> [B,N,1,K]
            torch.nn.Linear(self.num_fields, self.reduced_size, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(self.reduced_size, self.num_fields, bias=False),
            torch.nn.Sigmoid()
        )
    def forward(
        self, 
        x: torch.Tensor  # [B,N,D,K]  # K featuers, D feature dimensions
    ):
        squeezed_x = x.mean(dim=-2, keepdim=True)  # [B,N,D,K] -> [B,N,1,K]
        w = self.excitation(squeezed_x)  # [B,N,1,K]
        return torch.mul(x, w)  # [B,N,D,K]

        

class EmbeddingModule(torch.nn.Module):

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass


class LocalEmbeddingModule(EmbeddingModule):

    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb = torch.nn.Embedding(num_items + 1, item_embedding_dim, padding_idx=0)
        self.reset_params()

    def debug_str(self) -> str:
        return f"local_emb_d{self._item_embedding_dim}"

    def reset_params(self):
        for name, params in self.named_parameters():
            if '_item_emb' in name:
                if os.getenv('LOCAL_RANK', None) == 0:
                    print(f"Initialize {name} as truncated normal: {params.data.size()} params")
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                if os.getenv('LOCAL_RANK', None) == 0:
                    print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class LocalCrossFeatureEmbeddingModule(LocalEmbeddingModule):
    def __init__(
        self,
        num_items: int,
        # num_feature_values: dict,   # Sum of the number of possible values of each feature
        item_embedding_dim: int,
        use_user_attrs: bool = False,
	    projector: str = 'mean', # 'mean', 'concat'
    ) -> None:
        super().__init__(num_items, item_embedding_dim)

        self.projector_type = projector
        self.init_projector(self.projector_type)

    def init_projector(
        self,
        projector_type: str = 'concat',  # mean, concat
    ):
        self.senet = SENet(num_fields=self._multiplier, reduction_ratio=2)

        if projector_type == 'mean':
            self.projector = lambda x: x.mean(-1)

        elif projector_type == 'concat':
            self.projector = torch.nn.Sequential(  # DCN + MLP
                # LowRankCrossNet(in_features=self._item_embedding_dim*self._multiplier, num_layers=1, low_rank=self._item_embedding_dim//2), 
                torch.nn.Linear(self._item_embedding_dim * self._multiplier, self._item_embedding_dim, bias=False),    
            )
            # self.projector = torch.nn.Linear(self._item_embedding_dim * self._multiplier, self._item_embedding_dim, bias=False)

    def get_feature_embeddings(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([self._feature_emb[k](features[k]) for k in self._feat_keys], dim=-1)
    
    def get_user_attr_embeddings(self, user_attrs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([self._user_attr_emb[k](user_attrs[k]) for k in self._user_attr_keys], dim=-1)

    def get_behavior_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._item_emb(item_ids)

    def get_item_embeddings(
        self, 
        item_ids: torch.Tensor, 
        features: Dict = None, 
        user_attrs: Dict = None
    ) -> torch.Tensor:
        behavior_embeddings = self.get_behavior_embeddings(item_ids) # .squeeze()\
        output = behavior_embeddings  # [B,N,D]
        if features:
            feature_embeddings = self.get_feature_embeddings(features)  # [B,N,D,K]  # K distinct features fields
            stacked_embeddings = self.senet(torch.concat([output.unsqueeze(-1), feature_embeddings], dim=-1))  # [B,N,D,K+1]
            if self.projector_type == 'mean':
                # output = self.projector(torch.stack([output, self.projector(feature_embeddings)], dim=-1))
                output = self.projector(stacked_embeddings)
            elif self.projector_type == 'concat':
                output = self.projector(stacked_embeddings.flatten(-2,-1))  # [B,N,D,K+1] ---flatten---> [B,N,D*(K+1)] ---> [B,N,D]
        if user_attrs and self.use_user_attrs:
            user_attrs_embeddings = self.get_user_attr_embeddings(user_attrs).mean(-1)  # [B,1,D]
            # output = (user_attrs_embeddings + output) / 2
            output = torch.cat([user_attrs_embeddings, output[:,1:,...]], dim=1)  # [B,N,D]  # 拿掉最早的历史行为，替换为用户属性（继承最早行为的 timestamp）
        return output

class AddParameterModel(torch.nn.Module):
    def __init__(self, param_dim):
        super(AddParameterModel, self).__init__()
        self.learnable_param = torch.nn.Parameter(torch.randn(param_dim))
    
    def forward(self, x):
        return x + self.learnable_param
    
class ConcatenateEmbeddingModule(LocalEmbeddingModule):  # for irregular dimension settings
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        use_user_attrs: bool = False,
        attr_loss: bool = True,
        block_type = None,
        num_user_ids: int = 11675, # hash_id数量
        feat_meta_dict=None,
    ) -> None:
        """
        the final embedding = MLP(Concat(item_embedding+ feature_embeddings)) -> item_embedding_dim
        """
        super().__init__(num_items, item_embedding_dim)
            
        self._item_embedding_dim = item_embedding_dim
        self._attr_loss = attr_loss
        self._feat_keys = feat_meta_dict['seq_columns']
        self._feature_emb = torch.nn.ModuleDict({k: torch.nn.Embedding(feat_meta_dict['vocab_size'][k]+1, feat_meta_dict['emb_dim'][k], padding_idx=0) for k in self._feat_keys})
        feature_total_dim = sum([feat_meta_dict['emb_dim'][k] for k in self._feat_keys])
        self.input_dim = self._item_embedding_dim + feature_total_dim

        self.use_user_attrs = use_user_attrs
        self.block_type = block_type
        if 'mlp_idemb' in self.block_type:
            self.num_user_ids=num_user_ids
            self._user_id_dim= 8
            self._user_id_emb = torch.nn.Embedding(num_user_ids, self._user_id_dim)

        if self.use_user_attrs:
            self._user_attr_keys = feat_meta_dict['element_column']
            self._user_attr_emb = torch.nn.ModuleDict({k: torch.nn.Embedding(feat_meta_dict['vocab_size'][k]+1, feat_meta_dict['emb_dim'][k], padding_idx=0) for k in self._user_attr_keys})
            user_emb_dim = 4
            self.user_attr_dim = user_emb_dim
            self.user_attr_num = len(self._user_attr_emb)
            if 'contrast_useremb_pos' in self.block_type:
                self._user_attr_contrast_learning_factor=0.1
                user_attr_total_dim = sum([emb.embedding_dim for emb in self._user_attr_emb.values()])
                self.user_attr_projector = torch.nn.Linear(user_attr_total_dim, self.input_dim, bias=False)
            if 'concat_useremb' in block_type:
                user_attr_total_dim = sum([emb.embedding_dim for emb in self._user_attr_emb.values()])
                self.user_attr_projector = torch.nn.Linear(user_attr_total_dim, self.input_dim, bias=False)

    def get_user_id_embeddings(self, user_ids:List[str]):
        """将输入的uins列表转为分桶后的embedding
        Args:
            user_ids (list[str]): 长度为b的uins列表 其中每项为一个字符串
        Returns:
            user_id_embeddings: 形状为(b, _user_id_dim)的tensor
        """
        num_buckets=self.num_user_ids
        bucket_index_list = []
        for user_id in user_ids:
            bucket_index = self.get_bucket_index(user_id=user_id, num_buckets=num_buckets)
            bucket_index_list.append(bucket_index)
        # 将bucket_index_list转为tensor
        bucket_index_list = torch.tensor(bucket_index_list, dtype=torch.int32,device=next(self._user_id_emb.parameters()).device)
        # 获取user_id对应的embedding
        user_id_embeddings = self._user_id_emb(bucket_index_list)
        return user_id_embeddings

    def get_bucket_index(self,user_id, num_buckets):
        """单userid转bucket_index

        Args:
            user_id (str): 用户id
            num_buckets (int): 桶数

        Returns:
            int : 桶id 从0开始
        """
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        return hash_value % num_buckets  

    def get_feature_embeddings(self, features, feat_key=None):
        if feat_key is None:
            feature_embeddings = [(k, self._feature_emb[k](features[k])) for k in self._feat_keys] # [B,N,D']
            feature_embeddings = OrderedDict(feature_embeddings)
        else:
            feature_embeddings = self._feature_emb[feat_key](features)
        return feature_embeddings
    
    def get_user_attrs(self, user_attrs):
        user_attrs_embeddings = torch.concat([self._user_attr_emb[k](user_attrs[k]) for k in self._user_attr_keys], dim=-1)  # [B,N,D']
        return user_attrs_embeddings
    
    def get_item_embeddings(self, item_ids: torch.Tensor, features: Dict = None, user_attrs: Dict = None,return_feat_emb: bool = False, **kwargs) -> torch.Tensor:
        assert item_ids.max() < self._item_emb.weight.size(0)
        behavior_embeddings =self._item_emb(item_ids)  # [B,N,D]
        if features is not None:
            feature_embeddings = self.get_feature_embeddings(features) # [B,N,D']
            if return_feat_emb:
                return behavior_embeddings, feature_embeddings, features
            else:
                feature_embeddings = torch.concat([feature_embeddings[k] for k in self._feat_keys], dim=-1)  # [B,N,D']
                output = torch.concat([behavior_embeddings, feature_embeddings], dim=-1)  # [B,N,D+D']
        else:
            output = behavior_embeddings
            
        if user_attrs is not None and 'concat_useremb' in self.block_type:
            user_attrs_embeddings = self.get_user_attrs(user_attrs)
            # output = (user_attrs_embeddings + output) / 2
            output = torch.cat([user_attrs_embeddings, output[:,1:,...]], dim=1)  # [B,N,D]  # 拿掉最早的历史行为，替换为用户属性（继承最早行为的 timestamp
        if 'contrast_useremb_pos' in self.block_type  and user_attrs is not None:
            # 以位置编码的方式添加用户属性 常数
            user_attrs_embeddings = self.user_attr_projector(self.get_user_attrs(user_attrs))
            output = output * (1 - self._user_attr_contrast_learning_factor) + user_attrs_embeddings * self._user_attr_contrast_learning_factor
        return output


class CategoricalEmbeddingModule(EmbeddingModule):

    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        item_id_to_category_id: torch.Tensor,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb: torch.nn.Embedding = torch.nn.Embedding(num_items + 1, item_embedding_dim, padding_idx=0)  # ID2Embedding
        self.register_buffer("_item_id_to_category_id", item_id_to_category_id)
        self.reset_params()

    def debug_str(self) -> str:
        return f"cat_emb_d{self._item_embedding_dim}"

    def reset_params(self):
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(f"Initialize {name} as truncated normal: {params.data.size()} params")
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = self._item_id_to_category_id[(item_ids - 1).clamp(min=0)] + 1
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim

# To-Do: complete the feature embedding module
class FeatureEmbeddingModule(EmbeddingModule):
    
    def __init__(
        self, 
        item_embedding_dim: int,
        input_dim: int = 0
    ) -> None:
        super().__init__()
        self._item_embedding_dim: int = item_embedding_dim
        self._input_dim: int = input_dim
        self.reset_params()
    
    def debug_str(self) -> str:
        return f"feat_emb_d{self._item_embedding_dim}"
    
    def _init_encoder(self):
        self.encoder = torch.nn.Sequential()

    def reset_params(self):
        self._init_encoder()
        pass

    def get_item_embeddings(self, item_ids: torch.Tensor, features) -> torch.Tensor:
        # item_features = seq_features.past_payloads['features']
        item_features = features
        # item_ids = self._item_id_to_category_id[(item_ids - 1).clamp(min=0)] + 1
        if len(item_ids.size()) == 2:
            return torch.rand(item_ids.size(0), item_ids.size(1), self._item_embedding_dim).to(item_ids.device)
        else:
            return torch.rand(item_ids.size(0), self._item_embedding_dim).to(item_ids.device)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim
