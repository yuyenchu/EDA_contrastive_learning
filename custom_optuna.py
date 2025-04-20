from clearml.automation.optuna import OptimizerOptuna 
from clearml.automation.job import ClearmlJob
from logging import getLogger
from typing import (
    List,
    Set,
    Union,
    Sequence,
    Optional,
    Mapping,
    Callable,
    Tuple,
    Generator,
    Dict,
    Any,
)

logger = getLogger("clearml.automation.optimization")

class CustomOptimizerOptuna(OptimizerOptuna):
    def helper_create_job(
        self,
        base_task_id: str,
        parameter_override: Optional[Mapping[str, str]] = None,
        task_overrides: Optional[Mapping[str, str]] = None,
        tags: Optional[Sequence[str]] = None,
        parent: Optional[str] = None,
        **kwargs: Any
    ) -> ClearmlJob:
        if parameter_override:
            param_str = ["{}={}".format(k, parameter_override[k]) for k in sorted(parameter_override.keys())]
            if self._naming_function:
                name = self._naming_function(self._base_task_name, parameter_override)
            elif self._naming_function is False:
                name = None
            else:
                name = "{}: {}".format(self._base_task_name, " ".join(param_str))
            comment = "\n".join(param_str)
            augment_cfg = [(k,v) for k,v in parameter_override.items()]
            configuration_overrides = {
                'aug_cfg': {
                    'unlabeled_aug': {
                        'augment_cfg': augment_cfg,
                        'prob': 1,                    
                    },
                    'labeled_aug': {
                        'augment_cfg': augment_cfg,
                        'prob': 0.75,                    
                    }
                }
            }
        else:
            name = None
            comment = None
            configuration_overrides = None
        tags = (tags or []) + [
            self._tag,
            "opt" + (": {}".format(self._job_parent_id) if self._job_parent_id else ""),
        ]
        new_job = self._job_class(
            base_task_id=base_task_id,
            configuration_overrides=configuration_overrides,
            task_overrides=task_overrides,
            tags=tags,
            parent=parent or self._job_parent_id,
            name=name,
            comment=comment,
            project=self._job_project_id or self._get_task_project(parent or self._job_parent_id),
            **kwargs
        )
        self._created_jobs_ids[new_job.task_id()] = (new_job, parameter_override)
        logger.info("Creating new Task: {}".format(parameter_override))
        return new_job