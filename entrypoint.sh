#!/bin/sh
if [ -n "$AWS_LAMBDA_RUNTIME_API" ]; then
    exec python -m awslambdaric processor.handler.handler
else
    exec "$@"
fi
