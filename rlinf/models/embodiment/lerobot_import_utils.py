from __future__ import annotations

import inspect
from importlib import import_module, util
from pathlib import Path
import sys
import types


def _ensure_namespace_package(package_name: str, package_path: Path) -> None:
    module = sys.modules.get(package_name)
    if module is None:
        module = types.ModuleType(package_name)
        module.__package__ = package_name
        module.__path__ = [str(package_path)]
        spec = util.spec_from_loader(package_name, loader=None, is_package=True)
        if spec is not None:
            spec.submodule_search_locations = [str(package_path)]
            module.__spec__ = spec
        sys.modules[package_name] = module
        return

    module_path = getattr(module, "__path__", None)
    if module_path is None:
        module.__path__ = [str(package_path)]
    elif str(package_path) not in module_path:
        module_path.append(str(package_path))


def prepare_lerobot_policy_imports() -> None:
    """Avoid executing lerobot.policies.__init__ during submodule imports.

    Recent LeRobot builds eagerly import every policy from
    lerobot.policies.__init__, including GR00T, whose dataclass definitions are
    not compatible with Python 3.12. RLinf only needs targeted policy modules,
    so register lerobot.policies as a namespace package first and then import
    submodules directly.
    """
    lerobot = import_module("lerobot")
    policies_dir = Path(lerobot.__file__).resolve().parent / "policies"
    _ensure_namespace_package("lerobot.policies", policies_dir)


def _get_processor_import_parts(model_module: str) -> tuple[str, str]:
    module_prefix, module_name = model_module.rsplit(".", 1)
    policy_name = module_name.removeprefix("modeling_")
    processor_module = f"{module_prefix}.processor_{policy_name}"
    processor_symbol = f"make_{policy_name}_pre_post_processors"
    return processor_module, processor_symbol


def _wrap_processor_factory(processor_factory):
    signature = inspect.signature(processor_factory)
    accepted_params = set(signature.parameters)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    def make_pre_post_processors(policy_cfg, *args, **kwargs):
        if accepts_var_kwargs:
            return processor_factory(policy_cfg, *args, **kwargs)

        filtered_kwargs = {
            key: value for key, value in kwargs.items() if key in accepted_params
        }
        return processor_factory(policy_cfg, *args, **filtered_kwargs)

    return make_pre_post_processors


def import_lerobot_policy(model_module: str, model_symbol: str):
    """Import a LeRobot policy class and its processor factory lazily.

    Newer LeRobot builds expose a per-policy processor factory in each policy
    package, for example ``lerobot.policies.smolvla.processor_smolvla``.
    Importing those targeted modules avoids the global
    ``lerobot.policies.factory`` module, which eagerly imports unrelated policy
    configs such as Groot. Keep a factory fallback for older LeRobot revisions.
    """
    prepare_lerobot_policy_imports()
    policy_module = import_module(model_module)
    processor_module_name, processor_symbol = _get_processor_import_parts(
        model_module
    )

    try:
        processor_module = import_module(processor_module_name)
        make_pre_post_processors = _wrap_processor_factory(
            getattr(processor_module, processor_symbol)
        )
    except (ImportError, AttributeError):
        factory_module = import_module("lerobot.policies.factory")
        make_pre_post_processors = factory_module.make_pre_post_processors

    return make_pre_post_processors, getattr(policy_module, model_symbol)