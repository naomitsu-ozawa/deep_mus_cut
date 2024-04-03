#!/bin/bash

# 一時的に sudo のパスワードを無効化する
sudo -i << EOF

# vm.drop_caches の値を変更する
echo 3 > /proc/sys/vm/drop_caches

EOF