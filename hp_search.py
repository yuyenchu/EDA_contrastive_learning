import json

from clearml import Task
from clearml.automation import UniformParameterRange, UniformIntegerParameterRange, ParameterSet
from clearml.automation import HyperParameterOptimizer

from custom_optuna import CustomOptimizerOptuna

def get_aug_hp(aug_cfg, aug_name):
    for aug, defaults, hps in aug_cfg['augment_cfg']:
        if (aug==aug_name):
            return [
                UniformIntegerParameterRange(f'{aug}/{k}', v[0], v[1]) 
                if isinstance(v[0], int) and isinstance(v[1], int) else
                UniformParameterRange(f'{aug}/{k}', v[0], v[1])
                for k, v in hps.items()
            ]
    raise ValueError(f'Cannot find aug in augment_params.json: {aug_name}')

def job_complete_callback(
    job_id,                 # type: str
    objective_value,        # type: float
    objective_iteration,    # type: int
    job_parameters,         # type: dict
    top_performance_job_id  # type: str
):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('Record Broke! Objective reached {}'.format(objective_value))

if __name__=='__main__':
    task = Task.init(project_name='EDA_contrastive/hp_optimization',
                 task_name='Hyperparameter Search: EDA_contrastive',
                 task_type=Task.TaskTypes.optimizer,
                 tags=['optuna'],
                 reuse_last_task_id=True)
    configs = {
        'template_task_id': '4ba322b84e444d70abe9ecb6808ab9d3',
        'aug_target': ['GaussianNoise_Det'],
        'k': 3
    }
    configs = task.connect(configs)
    k=configs['k']

    with open('augment_params.json', 'r') as f:
        augment_cfg = json.load(f)
    hyper_parameters = []
    for aug_name in set(configs['aug_target']):
        hyper_parameters += get_aug_hp(augment_cfg, aug_name)

    optimizer = HyperParameterOptimizer(
        base_task_id=configs['template_task_id'],  # experiment to optimize
        # hyper-parameters to optimize
        hyper_parameters=hyper_parameters,
        # objective metric
        objective_metric_title='epoch_p_acc',
        objective_metric_series='epoch_p_acc',
        objective_metric_sign='max_global',

        # optimizer algorithm
        optimizer_class=CustomOptimizerOptuna,
        
        # params
        execution_queue='default', 
        max_number_of_concurrent_tasks=1, 
        optimization_time_limit=1440, # total time minutes
        compute_time_limit=90, # optimize compute time
        total_max_jobs=50,  
        save_top_k_tasks_only=k,
        min_iteration_per_job=20,
        max_iteration_per_job=400000
    )
    task.execute_remotely(queue_name="services", exit_process=True)

    optimizer.set_report_period(1) 
    optimizer.start(job_complete_callback=job_complete_callback)  
    optimizer.wait()

    top_exp = optimizer.get_top_experiments(top_k=k)
    print('Top {} experiments are:'.format(k))
    for n, t in enumerate(top_exp, 1):
        print('Rank {}: task id={} |result={}'
            .format(n, t.id, t.get_last_scalar_metrics()['epoch_p_acc']['epoch_p_acc']['max']))
        t.add_tags(f'Rank_{n}')
        cloned = Task.clone(t.id, project='EDA_contrastive', tags=['hp_search_result', f'hp:{task.id}', *t.get_tags()])
        # cloned.set_parameter('Args/epochs', 400, value_type=int)
        # Task.enqueue(cloned, queue_name='default')
    optimizer.stop()
    all_childs = [t for t in Task.get_tasks(project_name='Hyperparameter Optimization with BOHB', task_filter={'parent':task.task_id}) if t.task_id not in [i.id for i in top_exp]]
    for c in all_childs:
        c.delete_artifacts(list(c.artifacts.keys()))
    print('Optimization done')
    task.close()
