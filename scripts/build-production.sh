#!/bin/sh
set -e
pnpm --filter @workspace/api-server run build
pnpm --filter @workspace/proxy-ui run build
