#!/usr/bin/env bash

set -euo pipefail

# Download the prepared fineweb-edu dataset, as numpy files

mkdir fineweb-edu

path_config="/tmp/rclone.conf"
access_key_id="${CF_ACCESS_KEY_ID}"
secret_access_key="${CF_SECRET_ACCESS_KEY}"

cat << EOF > "${path_config}"
[cloudflare]
type = s3
provider = Cloudflare
access_key_id = ${access_key_id}
secret_access_key = ${secret_access_key}
endpoint = https://1bc6902063ebeda2e710e26bb7f25e08.r2.cloudflarestorage.com/training-datasets
acl = private
EOF

echo "Downloading val-0.npy"
rclone --config="${path_config}" copy "cloudflare:fineweb-edu/val-0.npy" fineweb-edu/ --progress

for i in {1..99}; do
    echo "Downloading train-${i}.npy"
    rclone --config="${path_config}" copy "cloudflare:fineweb-edu/train-${i}.npy" fineweb-edu/ --progress
done
