# Carla_Leaderboardv2_Challenge


## Install


git lfs clone

carla server 路径: 10.1.40.11:/mnt/disk01/panmingjie/carla

环境安装
```conda env create -f environment.yml --name TCP```
还需分别安装leaderboard和scenario文件夹下的requirements.txt



## Spin a Carla Server

Inside the installed carla folder, run
```sh CarlaUE4.sh -RenderOffscreen -nosound --world-port=2000```
注意world-port的指定，默认是2000，carla会占用指定port和port+1的端口

## Run Roach Data Collection
```sh leaderboard/scripts/run_evaluation.sh```



