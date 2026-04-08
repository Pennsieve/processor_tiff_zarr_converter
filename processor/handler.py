"""
Lambda handler that bridges the invocation payload to environment variables,
then runs the same processing logic as ECS mode.

Static env vars (PENNSIEVE_API_HOST, PENNSIEVE_API_HOST2, ENVIRONMENT, REGION,
DEPLOYMENT_MODE, and processor-specific params like INITIAL_DOWNSAMPLE, TILE_SIZE,
etc.) are set on the Lambda function configuration.

Per-invocation values (inputDir, outputDir, etc.) arrive in the event payload
and are exported as env vars here.
"""

import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
log = logging.getLogger("processor-ome-zarr")


def handler(event, context):
    log.info("Lambda handler invoked with event: %s", event)

    os.environ["INPUT_DIR"] = event.get("inputDir", "")
    os.environ["OUTPUT_DIR"] = event.get("outputDir", "")
    os.environ["WORKFLOW_INSTANCE_ID"] = event.get("workflowInstanceId", "")
    os.environ["SESSION_TOKEN"] = event.get("sessionToken", "")
    os.environ["REFRESH_TOKEN"] = event.get("refreshToken", "")

    _known_keys = {"inputDir", "outputDir", "workflowInstanceId", "sessionToken",
                   "refreshToken", "computeNodeId", "executionRunId",
                   "llmGovernorFunction"}
    for key, value in event.items():
        if key not in _known_keys and isinstance(value, str):
            os.environ[key] = value

    from processor.main import run
    run()

    return {"status": "success", "workflowInstanceId": event.get("workflowInstanceId", "")}
