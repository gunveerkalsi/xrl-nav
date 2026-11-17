from gymnasium.envs.registration import register

from .nav_env import ACTION_NAMES, DISCRETE_ACTIONS, SECTOR_NAMES, NavEnv  # noqa: F401

register(id="XRLNav-v0", entry_point="xrl_nav.envs.nav_env:NavEnv")
