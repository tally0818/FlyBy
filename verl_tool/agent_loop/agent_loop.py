# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import heapq
import logging
import os
import queue
import random
import threading
from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import Any, Optional

import hydra
import numpy as np
import ray
import torch
import uuid
from cachetools import LRUCache
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl_tool.trainer.ppo.reward import load_reward_manager
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr, rollout_trace_op
from verl.workers.rollout.replica import TokenOutput, get_rollout_replica_class
from tqdm.asyncio import tqdm

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AsyncLLMServerManager:
    'A class to manage multiple OpenAI compatible LLM servers. This class provides'

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], max_cache_size: int = 10000):
        'Initialize the AsyncLLMServerManager.'
        self.config = config
        self.server_handles = server_handles
        random.shuffle(self.server_handles)


        self.weighted_serveres = [[0, (hash(server), server)] for server in server_handles]
        heapq.heapify(self.weighted_serveres)


        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

    def _choose_server(self, request_id: str) -> ray.actor.ActorHandle:

        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        server = self.weighted_serveres[0][1][1]
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        self.request_id_to_server[request_id] = server
        return server

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        'Generate tokens from prompt ids.'
        server = self._choose_server(request_id)
        output = await server.generate.remote(
            request_id=str(uuid.uuid4()),
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            audio_data=audio_data,
        )
        return output


class AgentLoopMetrics(BaseModel):
    'Agent loop performance metrics.'

    generate_sequences: float = 0.0
    tool_calls: float = 0.0
    
    

class AgentLoopOutput(BaseModel):
    'Agent loop output.'

    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: Optional[list[float]] = None
    multi_modal_data: Optional[dict[str, Any]] = None
    reward_score: Optional[float] = None
    num_turns: int = 0
    metrics: AgentLoopMetrics
    extra_fields: dict[str, Any] = {}


class _InternalAgentLoopOutput(AgentLoopOutput):
    'Internal agent loop output with padded sequences.'

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    response_ids: torch.Tensor
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    response_mask: torch.Tensor
    attention_mask: torch.Tensor
    response_logprobs: Optional[torch.Tensor] = None
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    extra_fields: dict[str, Any] = {}



class _DummyConfig:
    def __init__(self, config: DictConfig) -> None:
        self.config = config


class AgentLoopBase(ABC):
    'An agent loop takes a input message, chat with OpenAI compatible LLM server and interact with various'

    _class_initialized = False

    def __init__(
        self,
        trainer_config: _DummyConfig,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        'Initialize agent loop, each sample will have its own loop instance.'
        self.init_class(config=trainer_config.config, tokenizer=tokenizer, processor=processor, **kwargs)
        self.config = trainer_config.config
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.loop = asyncio.get_running_loop()

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer, processor: AutoProcessor, **kwargs):
        "This is used to do heavy initialization work that should shared across all instances. It's only called once."
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        'Run agent loop to interact with LLM server and environment.'
        raise NotImplementedError


_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    'Register agent loop class.'

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


@ray.remote(num_cpus=1)
class BatchExecutor:
    'Batch executor is used to collect requests into a batch execution'

    def __init__(self, batch_func, micro_batch_size=1, max_batch_size=None):
        'Args:'
        self._q = queue.Queue()
        self._batch_func = batch_func
        self._max_batch = max_batch_size
        self._micro_batch_size = micro_batch_size

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    async def submit_task(self, item):
        'Blocking submission, returning Future'
        fut = Future()
        self._q.put((item, fut))
        async_fut = asyncio.wrap_future(fut)
        res = await async_fut
        return res

    def _worker_loop(self):
        while True:

            first, first_fut = self._q.get()
            items = [first]
            futs = [first_fut]


            while True:
                try:
                    next_item, next_fut = self._q.get_nowait()
                    items.append(next_item)
                    futs.append(next_fut)
                    if self._max_batch and len(items) >= self._max_batch:
                        break
                except queue.Empty:
                    while len(items) % self._micro_batch_size != 0:
                        next_item, next_fut = self._q.get()
                        items.append(next_item)
                        futs.append(next_fut)
                        if self._max_batch and len(items) >= self._max_batch:
                            break
                    break

            try:
                results = self._batch_func(items)
            except Exception as e:
                for f in futs:
                    f.set_exception(e)
            else:
                for f, r in zip(futs, results, strict=False):
                    f.set_result(r)


@ray.remote(num_cpus=1)
class RewardManagerWorker:
    'Reward manager worker to compute reward score asynchronously to overlap with agent loop.'

    def __init__(self, config: DictConfig, local_path: str, rm_executor: BatchExecutor = None) -> None:
        tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.reward_manager = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        self.rm_executor = rm_executor
        self.loop = asyncio.get_event_loop()

    async def compute_score(
        self,
        data: DataProto,
    ) -> dict:
        'Compute reward score for agent loop output.'
        result = await self.loop.run_in_executor(
            None,
            self.reward_wrapper,
            data,
            True,
        )

        reward_score = result["reward_tensor"].sum(dim=-1).item()
        reward_extra_info = {k: v[0] for k, v in result.get("reward_extra_info", {}).items()}
        return {"reward_score": reward_score, "reward_extra_info": reward_extra_info}

    def reward_wrapper(self, data: DataProto, return_dict=False) -> torch.Tensor:
        'Assemble reward functions and reward model into one function and expose it to the event loop'
        if self.rm_executor is not None:
            res = ray.get(self.rm_executor.submit_task.remote(data))
            data = data.union(res)

        return self.reward_manager(data, return_dict)


@ray.remote
class AgentLoopWorker:
    'Agent loop worker takes a batch of messages and run each message in an agent loop.'

    def __init__(
        self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], rm_executor: BatchExecutor = None
    ):
        'Initialize agent loop manager.'
        self.config = config
        self.server_manager = AsyncLLMServerManager(config, server_handles)
        self.rm_executor = rm_executor

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True)

        agent_loop_config_path = config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        if agent_loop_config_path:
            agent_loop_configs = OmegaConf.load(agent_loop_config_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.config.actor_rollout_ref.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.actor_rollout_ref.model.custom_chat_template
            self.tokenizer.chat_template = self.config.actor_rollout_ref.model.custom_chat_template

        self.reward_manager_worker = RewardManagerWorker.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            ),
        ).remote(self.config, local_path, self.rm_executor)

        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
        )
        

        run_time_context = ray.get_runtime_context()
        self.name = run_time_context.get_actor_name() or "unnamed"

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        'Generate sequences from agent loop.'
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )


        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature


        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )
        
        if self.config.actor_rollout_ref.agent.max_concurrent_trajectories is not None:
            print(f"Agent Worker {self.name} using semaphore with max concurrency {self.config.actor_rollout_ref.agent.max_concurrent_trajectories}")
            semaphore = asyncio.Semaphore(self.config.actor_rollout_ref.agent.max_concurrent_trajectories)
            def semaphore_wrapper(func):
                async def wrapper(*args, **kwargs):
                    async with semaphore:
                        return await func(*args, **kwargs)
                return wrapper
        else:
            def semaphore_wrapper(func):
                async def wrapper(*args, **kwargs):
                    return await func(*args, **kwargs)
                return wrapper

        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(asyncio.create_task(semaphore_wrapper(self._run_agent_loop)(sampling_params, trajectory_info[i], **kwargs)))

        print(f"Agent Worker {self.name} launching {len(tasks)} tasks...")
        outputs = await tqdm.gather(*tasks, desc=f"Agent Worker {self.name} Looping", total=len(tasks))
        print(f"Agent Worker {self.name} finished {len(tasks)} tasks.")
        output = self._postprocess(outputs)
        return output

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=_DummyConfig(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )
            kwargs["validate"] = trajectory["validate"]
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)




















            self.tokenizer.padding_side = "left"
            prompt_output = self.tokenizer.pad(
                {"input_ids": output.prompt_ids},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.prompt_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if prompt_output["input_ids"].dim() == 1:
                prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
                prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

            self.tokenizer.padding_side = "right"
            response_output = self.tokenizer.pad(
                {"input_ids": output.response_ids},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.response_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if response_output["input_ids"].dim() == 1:
                response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
                response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

            response_mask_output = self.tokenizer.pad(
                {"input_ids": output.response_mask},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.response_length,
                return_tensors="pt",
                return_attention_mask=False,
            )
            if response_mask_output["input_ids"].dim() == 1:
                response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

            response_logprobs = None
            if output.response_logprobs is not None:
                pad_size = self.config.actor_rollout_ref.rollout.response_length - len(output.response_logprobs)
                response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

            response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
            attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
            input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)





            multi_modal_inputs = None
            if (
                self.processor is not None 
                and "Qwen2_5OmniProcessor" in self.processor.__class__.__name__
            ):
                audios = output.multi_modal_data.get("audio", None)
                current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
                processor_kwargs = {"text": [current_text], "return_tensors": "pt"}
                if audios:
                    processor_kwargs["audio"] = audios
                multi_modal_inputs = self.processor(**processor_kwargs)
                multi_modal_inputs.pop("input_ids", None)
                multi_modal_inputs.pop("attention_mask", None)

                multi_modal_inputs = dict(multi_modal_inputs)

                position_ids = compute_position_id_with_mask(attention_mask)
            elif (
                self.processor is not None
                and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
            ):
                from verl.models.transformers.qwen2_vl import get_rope_index

                images = output.multi_modal_data.get("image", None)
                current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
                multi_modal_inputs = self.processor(text=[current_text], images=images, return_tensors="pt")
                multi_modal_inputs.pop("input_ids", None)
                multi_modal_inputs.pop("attention_mask", None)



                multi_modal_inputs = dict(multi_modal_inputs)

                image_grid_thw = multi_modal_inputs.get("image_grid_thw")
                video_grid_thw = multi_modal_inputs.get("video_grid_thw")
                second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts")

                vision_position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids.squeeze(0),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask.squeeze(0),
                ).unsqueeze(0)

                valid_mask = attention_mask[0].bool()
                text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
                text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
                text_position_ids = text_position_ids.unsqueeze(0)
                position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)
            else:
                position_ids = compute_position_id_with_mask(attention_mask)
            
            enable_async_reward = (
                self.rm_executor is not None and self.config.reward_model.enable_resource_pool
            ) or not self.config.reward_model.enable
            if output.reward_score is None and enable_async_reward:
                batch = TensorDict(
                    {
                        "prompts": prompt_output["input_ids"],
                        "responses": response_output["input_ids"],
                        "attention_mask": attention_mask,
                        "input_ids": input_ids,
                        "position_ids": position_ids,
                    },
                    batch_size=1,
                )
                non_tensor_batch = {
                    **{k: np.array([v]) for k, v in kwargs.items()},
                    "__num_turns__": np.array([output.num_turns]),
                }
                extra_fields = {}
                for key, val in output.extra_fields.items():
                    extra_fields[key] = np.array([val], dtype=object)

                non_tensor_batch.update(extra_fields)
                data = DataProto(
                    batch=batch,
                    non_tensor_batch=non_tensor_batch,
                    meta_info={"global_steps": trajectory["step"], "validate": trajectory["validate"]},
                )
                result = await self.reward_manager_worker.compute_score.remote(data)
                output.reward_score = result["reward_score"]
                output.extra_fields["reward_extra_info"] = result["reward_extra_info"]
                
            return _InternalAgentLoopOutput(
                prompt_ids=prompt_output["input_ids"],
                response_ids=response_output["input_ids"],
                input_ids=input_ids,
                position_ids=position_ids,
                response_mask=response_mask,
                attention_mask=attention_mask,
                response_logprobs=response_logprobs,
                multi_modal_inputs=multi_modal_inputs,
                multi_modal_data=output.multi_modal_data,
                reward_score=output.reward_score,
                num_turns=output.num_turns,
                metrics=output.metrics,
                extra_fields=output.extra_fields,
            )

    def _postprocess(self, inputs: list[_InternalAgentLoopOutput]) -> DataProto:
        'Process the padded outputs from _run_agent_loop and combine them into a batch.'

        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)

        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "response_mask": response_mask,
                "input_ids": input_ids,
                "attention_mask": attention_mask,

                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }


        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        reward_extra_keys.sort()
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])


        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]

        extra_fields = {}
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields)
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)
        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={"metrics": metrics, "reward_extra_keys": reward_extra_keys},
        )


async def get_trajectory_info(step, index, validate):
    'Get trajectory info.'
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class AgentLoopManager:
    'Agent loop manager that manages a group of agent loop workers.'

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup = None, rm_wg: RayWorkerGroup = None):
        'Initialize agent loop manager.'
        self.config = config
        self.worker_group = worker_group
        self.rm_executor = None
        self.rm_micro_batch_size = None
        if rm_wg:

            def batch_fn(data_list: list[DataProto]) -> list[torch.Tensor]:
                new_data_list = []
                for data in data_list:
                    temp_non_tensor_batch = {"__num_turns__": data.non_tensor_batch["__num_turns__"]}
                    temp_data = DataProto(batch=data.batch, non_tensor_batch=temp_non_tensor_batch)
                    new_data_list.append(temp_data)

                new_batch = DataProto.concat(new_data_list)
                out_data = rm_wg.compute_rm_score(new_batch)
                return out_data.split(1)

            self.rm_executor = BatchExecutor.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
            ).remote(batch_fn, rm_wg.world_size)

            self.rm_micro_batch_size = rm_wg.world_size

        self._initialize_llm_servers()
        self._init_agent_loop_workers()


        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

    def _initialize_llm_servers(self):
        rollout_world_size = (
            self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
            * self.config.actor_rollout_ref.rollout.data_parallel_size
            * self.config.actor_rollout_ref.rollout.pipeline_model_parallel_size
        )
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        )
        num_replicas = world_size // rollout_world_size

        rollout_replica_class = get_rollout_replica_class(self.config.actor_rollout_ref.rollout.name)
        rollout_config = self.config.actor_rollout_ref.rollout
        model_config = self.config.actor_rollout_ref.model
        self.rollout_replicas = [
            rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.trainer.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]
        if self.worker_group:
            self._run_all([server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        else:
            self._run_all([server.init_standalone() for server in self.rollout_replicas])
        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        expected_num_workers = self.config.trainer.nnodes
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers
        num_workers = max(expected_num_workers, num_workers)
        logger.warning(f"Initializing {num_workers} agent loop workers...")

        self.max_concurrent_trajectories = self.config.actor_rollout_ref.agent.get("max_concurrent_trajectories", None)
        if self.max_concurrent_trajectories is not None:

            self.config.actor_rollout_ref.agent.max_concurrent_trajectories = (
                self.max_concurrent_trajectories // num_workers
            )
            logger.warning(f"Per-Agent Loop Worker max concurrency: {self.config.actor_rollout_ref.agent.max_concurrent_trajectories}")

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):

            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                AgentLoopWorker.options(
                    name=f"agent_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(self.config, self.server_handles, self.rm_executor)
            )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        'Split input batch and dispatch to agent loop workers.'

        if self.rm_micro_batch_size and len(prompts) % self.rm_micro_batch_size != 0:
            raise ValueError(
                f"The length of prompts {len(prompts)} cannot divide the world size of rm_wg {self.rm_micro_batch_size}"
            )
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()
        print(f"Dispatching {len(prompts)} prompts to {len(self.agent_loop_workers)} agent loop workers...")
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        output = DataProto.concat(outputs)
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()


        metrics = [output.meta_info.pop("metrics") for output in outputs]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()


        slowest = np.argmax(t_generate_sequences + t_tool_calls)
        attention_mask = output.batch["attention_mask"][slowest]
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing

    def wake_up(self):
        'Wake up all rollout replica instances.'
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    def sleep(self):
        'Sleep all rollout replica instances.'
        self._run_all([replica.sleep() for replica in self.rollout_replicas])

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            await asyncio.gather(*tasks)

        asyncio.run(run_all())
