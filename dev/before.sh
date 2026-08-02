#!/usr/bin/env bash

# to run applications without sudo privileges, connect the glasses,
# make sure that permission to device files are set to 777 (it will be reverted after next login):

# TODO: should only be useful on Linux, need alternatives on Windows & MacOS

#echo "Please give sudo permission to open access to hidraw device(s) ..."
#sudo chmod 777 /dev/hidraw*
#echo "... permission granted:"
#ls -l /dev/hidraw*
