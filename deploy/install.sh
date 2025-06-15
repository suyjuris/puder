#!/bin/bash
set -eu

USER=puder
DIR=/opt/puder
DIR_RW="$DIR/cache $DIR/save"

useradd $USER || true

mkdir -p $DIR $DIR_RW
chown root: $DIR
chown $USER: $DIR_RW
install -g $USER -m 0440 -t "$DIR" *.py *.json 
install -g $USER -m 0550 -t "$DIR" deploy/start.sh

cp -r res tensors models .venv $DIR
chown -R root:$USER $DIR/tensors $DIR/models $DIR/.venv $DIR/res
chmod -R u=rX,g=rX,o= $DIR/tensors $DIR/models $DIR/.venv $DIR/res
chmod -R u=rx,g=rx,o= $DIR/.venv/bin

install -t /etc/systemd/system/ deploy/puder.service

systemctl daemon-reload
systemctl enable puder.service
