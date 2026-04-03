# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing as mp
import sys
from copy import deepcopy
from typing import Any, Callable, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import Env
from gymnasium.error import CustomSpaceError
from gymnasium.vector.async_vector_env import AsyncState, AsyncVectorEnv
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from gymnasium.vector.utils import (
    CloudpickleWrapper,
    clear_mpi_env_vars,
    concatenate,
    create_empty_array,
    create_shared_memory,
    iterate,
    read_from_shared_memory,
    write_to_shared_memory,
)
from gymnasium.vector.vector_env import VectorEnv
from numpy.typing import NDArray


class NoAutoResetSyncVectorEnv(SyncVectorEnv):
    def step(
        self, actions
    ) -> tuple[Any, NDArray[Any], NDArray[Any], NDArray[Any], dict]:
        """Steps through each of the environments returning the batched results.

        Returns:
            The batched environment step results
        """
        # gymnasium >=1.0 renamed _terminateds/_truncateds to _terminations/_truncations
        # and observations to _observations.  Support both old and new layouts.
        has_old_api = hasattr(self, "_terminateds")
        terminateds = self._terminateds if has_old_api else self._terminations
        truncateds = self._truncateds if has_old_api else self._truncations

        infos = {}
        observations = []
        for i, (env, action) in enumerate(zip(self.envs, iterate(self.action_space, actions))):
            (
                observation,
                self._rewards[i],
                terminateds[i],
                truncateds[i],
                info,
            ) = env.step(action)

            observations.append(observation)
            # Keep per-env obs list in sync (gymnasium >=1.0 uses _env_obs)
            if hasattr(self, "_env_obs"):
                self._env_obs[i] = observation
            infos = self._add_info(infos, info, i)

        # gymnasium >=1.0 uses _observations; older versions use observations
        if hasattr(self, "_observations"):
            self._observations = concatenate(
                self.single_observation_space, observations, self._observations
            )
            obs_out = self._observations
        else:
            self.observations = concatenate(
                self.single_observation_space, observations, self.observations
            )
            obs_out = self.observations

        return (
            deepcopy(obs_out) if self.copy else obs_out,
            np.copy(self._rewards),
            np.copy(terminateds),
            np.copy(truncateds),
            infos,
        )


def _worker_no_auto_reset(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is None
    env = env_fn()
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                observation, info = env.reset(**data)
                pipe.send(((observation, info), True))

            elif command == "step":
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(data)
                pipe.send(((observation, reward, terminated, truncated, info), True))
            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "seed", "close"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with "
                        f"`_call`. Use `{name}` directly instead."
                    )
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            elif command == "_check_spaces":
                pipe.send(
                    (
                        (data[0] == env.observation_space, data[1] == env.action_space),
                        True,
                    )
                )
            else:
                raise RuntimeError(
                    f"Received unknown command `{command}`. Must "
                    "be one of {`reset`, `step`, `seed`, `close`, `_call`, "
                    "`_setattr`, `_check_spaces`}."
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


def _worker_shared_memory_no_auto_reset(
    index, env_fn, pipe, parent_pipe, shared_memory, error_queue
):
    assert shared_memory is not None
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                observation, info = env.reset(**data)
                write_to_shared_memory(
                    observation_space, index, observation, shared_memory
                )
                pipe.send(((None, info), True))

            elif command == "step":
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(data)
                write_to_shared_memory(
                    observation_space, index, observation, shared_memory
                )
                pipe.send(((None, reward, terminated, truncated, info), True))
            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "seed", "close"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with "
                        f"`_call`. Use `{name}` directly instead."
                    )
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            elif command == "_check_spaces":
                pipe.send(
                    ((data[0] == observation_space, data[1] == env.action_space), True)
                )
            else:
                raise RuntimeError(
                    f"Received unknown command `{command}`. Must "
                    "be one of {`reset`, `step`, `seed`, `close`, `_call`, "
                    "`_setattr`, `_check_spaces`}."
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


class NoAutoResetAsyncVectorEnv(AsyncVectorEnv):
    """
    Rewrite the AsyncVectorEnv to disable auto reset when an env is done.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Env]],
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        shared_memory: bool = True,
        copy: bool = True,
        context: Optional[str] = None,
        daemon: bool = True,
        worker: Optional[Callable] = None,
    ):
        """
        Vectorized environment that runs multiple environments in parallel.

        Args:
            env_fns: Functions that create the environments.
            observation_space: Observation space of a single environment. If ``None``,
                then the observation space of the first environment is taken.
            action_space: Action space of a single environment. If ``None``,
                then the action space of the first environment is taken.
            shared_memory: If ``True``, then the observations from the worker processes are communicated back through
                shared variables. This can improve the efficiency if the observations are large (e.g. images).
            copy: If ``True``, then the :meth:`~AsyncVectorEnv.reset` and :meth:`~AsyncVectorEnv.step` methods
                return a copy of the observations.
            context: Context for `multiprocessing`_. If ``None``, then the default context is used.
            daemon: If ``True``, then subprocesses have ``daemon`` flag turned on; that is, they will quit if
                the head process quits. However, ``daemon=True`` prevents subprocesses to spawn children,
                so for some environments you may want to have it set to ``False``.
            worker: If set, then use that worker in a subprocess instead of a default one.
                Can be useful to override some inner vector env logic, for instance, how resets on termination or truncation are handled.

        Warnings:
            worker is an advanced mode option. It provides a high degree of flexibility and a high chance
            to shoot yourself in the foot; thus, if you are writing your own worker, it is recommended to start
            from the code for ``_worker`` (or ``_worker_shared_memory``) method, and add changes.
        """
        ctx = mp.get_context(context)
        self.env_fns = env_fns
        self.shared_memory = shared_memory
        self.copy = copy
        dummy_env = env_fns[0]()
        self.metadata = dummy_env.metadata

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or dummy_env.observation_space
            action_space = action_space or dummy_env.action_space
        dummy_env.close()
        del dummy_env
        VectorEnv.__init__(
            self=self,
            num_envs=len(env_fns),
            observation_space=observation_space,
            action_space=action_space,
        )

        if self.shared_memory:
            try:
                _obs_buffer = create_shared_memory(
                    self.single_observation_space, n=self.num_envs, ctx=ctx
                )
                self.observations = read_from_shared_memory(
                    self.single_observation_space, _obs_buffer, n=self.num_envs
                )
            except CustomSpaceError as e:
                raise ValueError(
                    "Using `shared_memory=True` in `AsyncVectorEnv` "
                    "is incompatible with non-standard Gymnasium observation spaces "
                    "(i.e. custom spaces inheriting from `gymnasium.Space`), and is "
                    "only compatible with default Gymnasium spaces (e.g. `Box`, "
                    "`Tuple`, `Dict`) for batching. Set `shared_memory=False` "
                    "if you use custom observation spaces."
                ) from e
        else:
            _obs_buffer = None
            self.observations = create_empty_array(
                self.single_observation_space, n=self.num_envs, fn=np.zeros
            )

        self.parent_pipes, self.processes = [], []
        self.error_queue = ctx.Queue()
        target = (
            _worker_shared_memory_no_auto_reset
            if self.shared_memory
            else _worker_no_auto_reset
        )
        target = worker or target
        with clear_mpi_env_vars():
            for idx, env_fn in enumerate(self.env_fns):
                parent_pipe, child_pipe = ctx.Pipe()
                process = ctx.Process(
                    target=target,
                    name=f"Worker<{type(self).__name__}>-{idx}",
                    args=(
                        idx,
                        CloudpickleWrapper(env_fn),
                        child_pipe,
                        parent_pipe,
                        _obs_buffer,
                        self.error_queue,
                    ),
                )

                self.parent_pipes.append(parent_pipe)
                self.processes.append(process)

                process.daemon = daemon
                process.start()
                child_pipe.close()

        self._state = AsyncState.DEFAULT
        self._check_spaces()
