#!/bin/bash
  spark-submit  --master local $1.py &>out &


  audria $(ps -aux | grep -v grep | grep yl408 | grep spark-2.3.2 | awk '{print $2}')  -d 1 >result_$1.csv &
 
  sudo ~/scripts/bash_scripts/IntelPerformanceCounterMonitorV2.8/pcm.x -r -csv=pcm_result_$1.csv -i=180 &
