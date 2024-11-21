"""accelerate dataclasses"""
import enum
import copy
import functools
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union, get_args

from mindnlp.accelerate.utils.config import (
    MindformersTrainningConfig,
    MindFormersModelParallelConfig,
    MindForemrsOptimizerConfig,
    MindFormersTransformerConfig
)

class EnumWithContains(enum.EnumMeta):
    "A metaclass that adds the ability to check if `self` contains an item with the `in` operator"

    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True
    
class BaseEnum(enum.Enum, metaclass=EnumWithContains):
    "An enum class that can get the value of an item with `str(Enum.key)`"

    def __str__(self):
        return self.value

    @classmethod
    def list(cls):
        "Method to list all the possible items in `cls`"
        return list(map(str, cls))
    
class DDPCommunicationHookType(BaseEnum):
    """
    Represents a type of communication hook used in DDP.

    Values:

        - **NO** -- no communication hook
        - **FP16** -- DDP communication hook to compress the gradients in FP16
        - **BF16** -- DDP communication hook to compress the gradients in BF16
        - **POWER_SGD** -- DDP communication hook to use PowerSGD
        - **BATCHED_POWER_SGD** -- DDP communication hook to use batched PowerSGD
    """

    NO = "no"
    FP16 = "fp16"
    BF16 = "bf16"
    POWER_SGD = "power_sgd"
    BATCHED_POWER_SGD = "batched_power_sgd"


class KwargsHandler:
    """
    Internal mixin that implements a `to_kwargs()` method for a dataclass.
    """

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_kwargs(self):
        """
        Returns a dictionary containing the attributes with values different from the default of this class.
        """
        # import clear_environment here to avoid circular import problem
        from .other import clear_environment

        with clear_environment():
            default_dict = self.__class__().to_dict()
        this_dict = self.to_dict()
        return {k: v for k, v in this_dict.items() if default_dict[k] != v}

@dataclass
class DistributedDataParallelKwargs(KwargsHandler):
    """
    Use this object in your [`Accelerator`] to customize how your model is wrapped in a
    `torch.nn.parallel.DistributedDataParallel`. Please refer to the documentation of this
    [wrapper](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) for more
    information on each argument.

    <Tip warning={true}>

    `gradient_as_bucket_view` is only available in PyTorch 1.7.0 and later versions.

    `static_graph` is only available in PyTorch 1.11.0 and later versions.

    </Tip>

    Example:

    ```python
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    ```
    """

    dim: int = 0
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25
    find_unused_parameters: bool = False
    check_reduction: bool = False
    gradient_as_bucket_view: bool = False
    static_graph: bool = False

    comm_hook: DDPCommunicationHookType = DDPCommunicationHookType.NO
    comm_wrapper: Literal[
        DDPCommunicationHookType.NO, DDPCommunicationHookType.FP16, DDPCommunicationHookType.BF16
    ] = DDPCommunicationHookType.NO
    comm_state_option: dict = field(default_factory=dict)

    def to_dict(self, ignore_keys=("comm_hook", "comm_wrapper", "comm_state_option")):
        return {k: v for k, v in super().to_dict().items() if k not in ignore_keys}

    def register_comm_hook(self, model):
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks, powerSGD_hook

        hook_map: Dict[DDPCommunicationHookType, Callable] = {
            DDPCommunicationHookType.FP16: default_hooks.fp16_compress_hook,
            DDPCommunicationHookType.BF16: default_hooks.bf16_compress_hook,
            DDPCommunicationHookType.POWER_SGD: powerSGD_hook.powerSGD_hook,
            DDPCommunicationHookType.BATCHED_POWER_SGD: powerSGD_hook.batched_powerSGD_hook,
        }

        wrapper_map: Dict[DDPCommunicationHookType, Callable] = {
            DDPCommunicationHookType.FP16: default_hooks.fp16_compress_wrapper,
            DDPCommunicationHookType.BF16: default_hooks.bf16_compress_wrapper,
        }

        hook: Optional[Callable] = hook_map.get(self.comm_hook)
        wrapper: Optional[Callable] = wrapper_map.get(self.comm_wrapper)

        if hook and wrapper:
            hook = wrapper(hook)

        if hook:
            state = (
                powerSGD_hook.PowerSGDState(None, **self.comm_state_option)
                if self.comm_hook in (DDPCommunicationHookType.POWER_SGD, DDPCommunicationHookType.BATCHED_POWER_SGD)
                else None
            )
            model.register_comm_hook(
                state=state,
                hook=hook,
            )


class DistributedType(str, enum.Enum):
    """
    Represents a type of distributed environment.

    Values:
        - **MINDFORMERS** -- Using mindformers
    """

    MINDFORMERS = "MINDFORMERS"
    NO = "NO"


@dataclass
class MindFormersPlugin:
    """
    Plugin for MindFormersLM to enable tensor, pipeline, sequence and data parallelism.
    """

    def __post_init__(self):
        self.mindformers_default_args = {
            "trainning_config": {},
            "parallel_config": {},
            "model_config": {},
            "dataset_config": {},
            "optimizer_config": {}
        }

    def set_trainning_args(self):
        trainning_config = MindformersTrainningConfig()
        self.mindformers_default_args["trainning_config"] = asdict(trainning_config)

    def set_optimizer_args(self):
        optimizer_config = MindForemrsOptimizerConfig()
        self.mindformers_default_args["optimizer_config"] = asdict(optimizer_config)

    def set_paralle_args(self):
        parallel_config = MindFormersModelParallelConfig()
        self.mindformers_default_args["parallel_config"] = asdict(parallel_config)

    def set_model_args(self, model, batch_data):
        model_config_type = model.config.model_type.lower()
        MODEL_CONFIGS_TO_MINDFORMERS_PARSERS[model_config_type](self, model, batch_data)

    @property
    def config_dict(self):
        return self.mindformers_default_args

    @property
    def model_type(self):
        model_type = "llama"
        return model_type


MODEL_CONFIGS_TO_MINDFORMERS_PARSERS = {}


def add_model_config_to_mindformers_parser(model_type: str):
    def add_model_config_parser_helper(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        MODEL_CONFIGS_TO_MINDFORMERS_PARSERS[model_type] = func
        return wrapper

    return add_model_config_parser_helper


@add_model_config_to_mindformers_parser("llama")
def parse_llama_config(mindformers_plugin, model, batch_data):
    model_config = MindFormersTransformerConfig(
        vocab_size=1200,
        hidden_size=128,
        ffn_hidden_size=512,
        num_layers=2,
        num_heads=8,
    )
    mindformers_plugin.mindformers_default_args["model_config"] = asdict(model_config)
