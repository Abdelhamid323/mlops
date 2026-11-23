from prefect import serve
from prefect.client.schemas.schedules import CronSchedule

from pipeline_prefect import flow_all, flow_code, flow_evaluate, flow_train

TZ = "Africa/Tunis"

if __name__ == "__main__":
    all_dep = flow_all.to_deployment(
        name="ml-pipeline-all",
        schedules=[CronSchedule(cron="0 2 * * *", timezone=TZ)],
        tags=["mlops", "full-pipeline"],
    )
    train_dep = flow_train.to_deployment(
        name="ml-pipeline-train",
        schedules=[CronSchedule(cron="0 3 * * *", timezone=TZ)],
        tags=["mlops", "train"],
    )
    evaluate_dep = flow_evaluate.to_deployment(
        name="ml-pipeline-evaluate",
        schedules=[CronSchedule(cron="0 8 * * *", timezone=TZ)],
        tags=["mlops", "evaluate"],
    )
    code_dep = flow_code.to_deployment(name="ml-pipeline-code", tags=["mlops", "ci"])

    serve(all_dep, train_dep, evaluate_dep, code_dep)