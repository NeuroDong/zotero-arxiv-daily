import os
import re
import sys
import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from loguru import logger
import dotenv
from zotero_arxiv_daily.executor import Executor
os.environ["TOKENIZERS_PARALLELISM"] = "false"
dotenv.load_dotenv()

# Match ${oc.env:VAR} or ${oc.env:VAR,default} (default can be "null" or any string)
_OC_ENV_PATTERN = re.compile(r"^\$\{oc\.env:([^,}]+)(?:,([^}]*))?\}$")


def _resolve_oc_env_string(s: str) -> str | None:
    """If s is literal '${oc.env:VAR}' or '${oc.env:VAR,default}', return env value."""
    if not isinstance(s, str):
        return None
    m = _OC_ENV_PATTERN.match(s.strip())
    if not m:
        return None
    var_name = m.group(1).strip()
    default = m.group(2)
    if default is not None:
        default = default.strip()
        if default == "null":
            default = None
    return os.environ.get(var_name, default)


def _apply_env_to_container(obj):
    """Recursively replace literal ${oc.env:...} strings in dict/list with env values."""
    if isinstance(obj, dict):
        return {k: _apply_env_to_container(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_apply_env_to_container(v) for v in obj]
    if isinstance(obj, str):
        resolved = _resolve_oc_env_string(obj)
        if resolved is not None:
            return resolved
    return obj


def _ensure_env_resolved(config: DictConfig) -> DictConfig:
    """Resolve interpolations, then replace any remaining literal ${oc.env:...} from os.environ."""
    try:
        OmegaConf.resolve(config)
    except Exception:
        pass
    # If values are plain strings (e.g. from merged YAML), resolve() won't change them.
    # Convert to container and replace literal oc.env strings, then re-build config.
    container = OmegaConf.to_container(config, resolve=False)
    resolved_container = _apply_env_to_container(container)
    return OmegaConf.create(resolved_container)


@hydra.main(version_base=None, config_path="../../config", config_name="default")
def main(config: DictConfig):
    config = _ensure_env_resolved(config)

    # Configure loguru log level based on config
    log_level = "DEBUG" if config.executor.debug else "INFO"
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    for logger_name in logging.root.manager.loggerDict:
        if "zotero_arxiv_daily" in logger_name:
            continue
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    if config.executor.debug:
        logger.info("Debug mode is enabled")
    
    executor = Executor(config)
    executor.run()

if __name__ == '__main__':
    main()