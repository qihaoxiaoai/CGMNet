# cgmnet/utils/register.py
"""
Implements a registry pattern for models, datasets, and collators.

This allows for a flexible and modular project structure, where components can be
added and then accessed from main scripts using simple string keys.
"""
from typing import Callable, Dict, Type, Any

# --- Global Dictionaries for Registration ---
MODEL_REGISTRY: Dict[str, Type] = {}
DATASET_REGISTRY: Dict[str, Type] = {}
COLLATOR_REGISTRY: Dict[str, Type] = {}


# --- Decorator Functions ---

def register_model(name: str) -> Callable:
    """
    A decorator to register a new model class.

    Example:
        @register_model("cgmnet")
        class CGMNet(nn.Module):
            ...

    Args:
        name: The string key to register the model under.

    Returns:
        The decorator function.
    """
    def decorator(model_cls: Type) -> Type:
        MODEL_REGISTRY[name] = model_cls
        return model_cls
    return decorator


def register_dataset(name: str) -> Callable:
    """
    A decorator to register a new dataset class.

    Example:
        @register_dataset("cgmnet_finetune")
        class CGMNetFinetuneDataset(Dataset):
            ...

    Args:
        name: The string key to register the dataset under.

    Returns:
        The decorator function.
    """
    def decorator(dataset_cls: Type) -> Type:
        DATASET_REGISTRY[name] = dataset_cls
        return dataset_cls
    return decorator


def register_collator(name: str) -> Callable:
    """
    A decorator to register a new collator class.

    Example:
        @register_collator("cgmnet_finetune")
        class CGMNetFinetuneCollator:
            ...

    Args:
        name: The string key to register the collator under.

    Returns:
        The decorator function.
    """
    def decorator(collator_cls: Type) -> Type:
        COLLATOR_REGISTRY[name] = collator_cls
        return collator_cls
    return decorator
