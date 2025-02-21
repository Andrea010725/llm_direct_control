#!/bin/bash
export CARLA_ROOT=/home/wupenghao/transfuser/carla_lb2
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:/home/wupenghao/transfuser/Carla_Leaderboardv2_Challenge/leaderboard
export PYTHONPATH=$PYTHONPATH:/home/wupenghao/transfuser/Carla_Leaderboardv2_Challenge/leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:/home/wupenghao/transfuser/Carla_Leaderboardv2_Challenge/scenario_runner


export LEADERBOARD_ROOT=/home/wupenghao/transfuser/Carla_Leaderboardv2_Challenge/leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export SCENARIO_RUNNER_ROOT=/home/wupenghao/transfuser/Carla_Leaderboardv2_Challenge/scenario_runner
export PORT=2000 # same as the carla server port
export TM_PORT=8000 # port for traffic manager, required when spawning multiple servers/clients
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs

export RESUME=True
# export IGNORE_SCENARIO=True
export EVAL=True
# export DATAGEN=1


# autopilot
# export ROUTES=/home/wupenghao/transfuser/Carla_Leaderboardv2_Challenge/leaderboard/data/lane_changing_cases/Accident.xml
# export TEAM_AGENT=team_code_autopilot/autopilot.py
# export TEAM_CONFIG=/home/wupenghao/transfuser/Carla_Leaderboardv2_Challenge/roach/config/config_agent.yaml
# export CHECKPOINT_ENDPOINT=/home/wupenghao/transfuser/Carla_Leaderboardv2_Challenge/Accident.json
# export SAVE_PATH=/home/wupenghao/transfuser/data_roach_90/Accident # path 


# eval TCP
# export ROUTES=/home/wupenghao/transfuser/Carla_Leaderboardv2_Challenge/leaderboard/data/ParkingExit_cases/ParkingExit.xml
# export TEAM_AGENT=team_code/tcp_agent.py
# export TEAM_CONFIG=/home/wupenghao/transfuser/Carla_Leaderboardv2_Challenge/TCP/log/lb2_90routes_rgbhigh_half/epoch=59-last.ckpt
# export CHECKPOINT_ENDPOINT=eval/TCP_ParkingExit.json
# export SAVE_PATH=eval/TCP_ParkingExit # path 

# roach
# export ROUTES=/home/wupenghao/transfuser/Carla_Leaderboardv2_Challenge/leaderboard/data/routes_training_split_5.xml
# export TEAM_AGENT=team_code/roach_ap_agent.py
# export TEAM_CONFIG=/home/wupenghao/transfuser/Carla_Leaderboardv2_Challenge/roach/config/config_agent.yaml
# export CHECKPOINT_ENDPOINT=routes_training_split_5.json
# # export DEBUG_CHECKPOINT_ENDPOINT=roach_ap_90routes_5_result_debug.json
# export SAVE_PATH=/home/wupenghao/transfuser/data_roach_90/routes_training_split_5 # path 

# eval aim v2
export ROUTES=/home/wupenghao/transfuser/Carla_Leaderboardv2_Challenge/leaderboard/data/ParkingExit_cases/ParkingExit.xml
export TEAM_AGENT=team_code/aim_v2_agent.py
export TEAM_CONFIG=/home/wupenghao/transfuser/Carla_Leaderboardv2_Challenge/aim_v2/log/lb2_90routes_rgbhigh_half/epoch=59-last.ckpt
export CHECKPOINT_ENDPOINT=eval/aim_v2_ParkingExit.json
export SAVE_PATH=eval/aim_v2_ParkingExit # path 


python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--routes=${ROUTES} \
--routes-subset=${ROUTES_SUBSET} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--debug-checkpoint=${DEBUG_CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT}

