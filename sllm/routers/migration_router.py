# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
from typing import Dict, Optional

import ray

from sllm.inference_instance import start_instance
from sllm.logger import init_logger
from sllm.routers.roundrobin_router import RoundRobinRouter
from sllm.utils import InstanceHandle, InstanceStatus

logger = init_logger(__name__)


class MigrationRouter(RoundRobinRouter):
    def __init__(
        self,
        model_name: str,
        resource_requirements: Dict[str, int],
        backend: str,
        backend_config: Dict,
        router_config: Dict,
        enable_lora: bool = False,
        lora_adapters: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(
            model_name,
            resource_requirements,
            backend,
            backend_config,
            router_config,
            enable_lora,
            lora_adapters,
        )
        self.migration_record = {}
        self.migration_failures = {}
        self.migration_delta = self.router_config.get("migration_delta", 20)

    async def inference(self, request_data: dict, action: str):
        async with self.running_lock:
            if not self.running:
                return {"error": "Instance stopped"}

        async with self.request_count_lock:
            self.request_count += 1

        async with self.idle_time_lock:
            self.idle_time = 0

        instance = None
        try:
            instance_allocation = self.loop.create_future()
            await self.request_queue.put(instance_allocation)
            instance_id = await instance_allocation
            async with self.instance_management_lock:
                instance = self.ready_inference_instances.get(instance_id)
            if instance is None:
                return {"error": "Instance not found"}

            if self.enable_lora and "lora_adapter_name" in request_data:
                adapter = request_data["lora_adapter_name"]
                if adapter not in self.loaded_lora_adapters:
                    return {"error": f"Lora adapter {adapter} not found"}
                await instance.backend_instance.load_lora_adapter.remote(
                    lora_name=adapter,
                    lora_path=self.loaded_lora_adapters[adapter],
                )
            if action == "generate":
                result = await instance.backend_instance.generate.remote(
                    request_data=request_data
                )
                if "preempted" in result:
                    if self.enable_lora or request_data.get("lora_adapter_name"):
                        return {"error": "LoRA migration is not supported"}
                    async with self.instance_management_lock:
                        target_id = self.migration_record.get(instance_id)
                        target = self.ready_inference_instances.get(target_id)
                    if target is None:
                        return {"error": "No target instance found"}
                    # The backend computes the remaining budget from the full
                    # source output, including when max_tokens was omitted.
                    result = await target.backend_instance.resume_generate.remote(
                        request_data=request_data,
                        current_output=result["current_output"],
                    )
            elif action == "encode":
                result = await instance.backend_instance.encode.remote(
                    request_data=request_data
                )
            else:
                result = {"error": "Invalid action"}
            return result
        finally:
            if instance is not None:
                await instance.add_requests(-1)
            async with self.request_count_lock:
                self.request_count -= 1

    async def _retire_migration_instance(self, target_instance):
        target_instance.ready = False
        instance_id = target_instance.instance_id
        self.deleting_inference_instances[instance_id] = target_instance
        shutdown_error = None
        if target_instance.backend_instance is not None:
            try:
                await target_instance.backend_instance.shutdown.remote()
            except Exception as error:
                shutdown_error = error
                logger.exception("Shutdown failed for %s; attempting kill", instance_id)
            # Do not release a reservation when actor termination is uncertain.
            # Keep the handle in the deleting pool for explicit recovery.
            ray.kill(target_instance.backend_instance)
        await self.model_loading_scheduler.deallocate_resource.remote(
            self.model_name, instance_id, self.resource_requirements
        )
        self.deleting_inference_instances.pop(instance_id, None)
        if shutdown_error is not None:
            raise shutdown_error

    async def execute_migration_plan(self, migration_plan):
        logger.info(f"Executing migration plan: {migration_plan}")
        if self.enable_lora:
            logger.error("LoRA migration is not supported")
            return None
        source_instance = migration_plan.source_instance
        source_instance_id = source_instance.instance_id
        target_node_id = migration_plan.target_node_id
        # start the target_instance on the target node
        startup_config = {
            "num_cpus": self.resource_requirements["num_cpus"],
            "num_gpus": self.resource_requirements["num_gpus"],
            "resources": {
                "worker_node": 0.1,
                f"worker_id_{target_node_id}": 0.1,
            },
        }
        logger.info(f"Startup config: {startup_config}, {self.backend_config}")

        instance_id = self._new_instance_id()
        logger.info(
            f"Creating new instance {instance_id} for model {self.model_name}"
        )
        # TODO: Add max_queue_length to target_instance
        target_instance = InstanceHandle(
            instance_id=instance_id,
            max_queue_length=10,
            num_gpu=self.resource_requirements["num_gpus"],
        )

        ret = await self.model_loading_scheduler.mark_resource.remote(
            self.model_name, instance_id, target_node_id
        )
        if not ret:
            logger.error(
                f"Failed to mark resource for instance {instance_id} for model {self.model_name}"
            )
            return None

        try:
            await start_instance.options(
                resources={
                    "worker_node": 0.1,
                    f"worker_id_{target_node_id}": 0.1,
                }
            ).remote(
                instance_id,
                self.backend,
                self.model_name,
                self.backend_config,
                startup_config,
            )
            logger.info(
                f"Started instance {instance_id} for model {self.model_name}"
            )
            target_instance.backend_instance = ray.get_actor(instance_id)
            async with target_instance.lock:
                target_instance.ready = True
                target_instance.node_id = target_node_id
            logger.info(
                f"Initialized instance {instance_id} for model {self.model_name}"
            )
            await target_instance.backend_instance.init_backend.remote()
            logger.info(
                f"Initialized backend for instance {instance_id} for model {self.model_name}"
            )
            # migrate the tokens from the source_instance (if still running) to the target_instance
            if source_instance_id not in self.ready_inference_instances:
                logger.info(f"Instance {source_instance_id} not found")
                raise RuntimeError(f"Source instance {source_instance_id} disappeared")
            source_instance = self.ready_inference_instances[source_instance_id]
            migration_iter = 0
            n_previous_tokens = 0
            while True:
                logger.info(f"Migration iteration {migration_iter}")
                current_tokens = (
                    await source_instance.backend_instance.get_current_tokens.remote()
                )
                if not current_tokens:
                    logger.info("No tokens found")
                    break
                n_current_tokens = len(current_tokens[0])
                n_delta_tokens = n_current_tokens - n_previous_tokens
                logger.info(
                    f"Number of tokens: {n_current_tokens}, delta: {n_delta_tokens}"
                )
                n_previous_tokens = n_current_tokens
                if n_delta_tokens <= self.migration_delta:
                    logger.info(
                        "Migration completed: remained "
                        f"{None if not current_tokens else n_delta_tokens} tokens"
                    )
                    break
                await target_instance.backend_instance.resume_kv_cache.remote(
                    current_tokens
                )
                logger.info(f"Migration iteration {migration_iter} completed")
                migration_iter += 1

            logger.info(f"Migrated instance {source_instance_id} to {instance_id}")
            async with self.instance_management_lock:
                if source_instance_id not in self.ready_inference_instances:
                    # source_instance has been removed
                    logger.error(f"Instance {instance_id} not found")
                    raise RuntimeError(f"Source instance {source_instance_id} disappeared")
                _ = self.ready_inference_instances.pop(source_instance_id)
                async with source_instance.lock:
                    source_instance.status = False
                self.ready_inference_instances[instance_id] = target_instance
                self.migration_record[source_instance_id] = instance_id
        except Exception:
            logger.exception("Failed to prepare migration target %s", instance_id)
            await self._retire_migration_instance(target_instance)
            return None
        logger.info(f"Instance {source_instance_id} removed")
        try:
            await self._retire_migration_instance(source_instance)
        except Exception as retirement_error:
            self.migration_failures[source_instance_id] = str(retirement_error)
            async with self.instance_management_lock:
                self.migration_record.pop(source_instance_id, None)
                self.ready_inference_instances.pop(instance_id, None)
            # A failed retirement is not a successful migration. The source may
            # already be stopped, so never put it back in the ready pool.
            try:
                await self._retire_migration_instance(target_instance)
            except Exception as cleanup_error:
                self.migration_failures[instance_id] = str(cleanup_error)
                raise retirement_error from cleanup_error
            raise
        return instance_id

    async def get_instance_status(self, instance_id: str) -> InstanceStatus:
        logger.info(f"Getting status for instance: {instance_id}")
        async with self.instance_management_lock:
            if instance_id not in self.ready_inference_instances:
                logger.info(f"Instance {instance_id} not found")
                return None
            instance = self.ready_inference_instances[instance_id]
            logger.info(f"Instance: {instance}")
            instance_status = await instance.get_status()
            instance_status.model_name = self.model_name
            current_tokens = (
                await instance.backend_instance.get_current_tokens.remote()
            )
            if current_tokens:
                instance_status.num_current_tokens = len(current_tokens[0])
            else:
                instance_status.num_current_tokens = 0
            logger.info(f"Instance status: {instance_status}")
            return instance_status

    async def fine_tuning(self, request_data: dict):
        raise NotImplementedError(
            "Fine-tuning is only applicable on roundrobin router"
        )
