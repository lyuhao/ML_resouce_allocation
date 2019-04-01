#!/bin/bash
  spark-submit  --master  spark://10.148.54.61:7077 $1.py &


  #audria $(ps -aux | grep -v grep | grep yl408 | grep spark-2.3.2 | awk '{print $2}')  -d 1 >result_$1.csv &
 
  #sudo ~/scripts/bash_scripts/IntelPerformanceCounterMonitorV2.8/pcm.x -r -csv=pcm_result_$1.csv -i=180 &
