#!/usr/bin/env python

# Copyright (c) 2018-2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module contains the result gatherer and write for CARLA scenarios.
It shall be used from the ScenarioManager only.
"""

from __future__ import print_function

import time
from collections import OrderedDict
from tabulate import tabulate

from easydict import EasyDict as Edict
import json
import os
# import ipdb

COLORED_STATUS = {
    "FAILURE": '\033[91mFAILURE\033[0m',
    "SUCCESS": '\033[92mSUCCESS\033[0m',
    "ACCEPTABLE": '\033[93mACCEPTABLE\033[0m',
}

STATUS_PRIORITY = {
    "FAILURE": 0,
    "ACCEPTABLE": 1,
    "SUCCESS": 2,
}  # Lower number is higher priority


class ResultOutputProvider(object):

    """
    This module contains the _result gatherer and write for CARLA scenarios.
    It shall be used from the ScenarioManager only.
    """

    def __init__(self, data, global_result, town, casetype_town12=None):
        """
        - data contains all scenario-related information
        - global_result is overall pass/fail info
        """
        self._data = data
        self._global_result = global_result
        self._casetype_town12 = casetype_town12

        self._start_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                         time.localtime(self._data.start_system_time))
        self._end_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                       time.localtime(self._data.end_system_time))

        print(self.create_output_text(town))

    def get_data(self):
        return self._data

    def get_data_dir(self):
        return self.data_dir



    def create_output_text(self, town):
        """
        Creates the output message
        """

        data = {}
        data_dir = ''
        town = town.lower().replace('hd', '')
        # dir_root = "/mnt/afs/user/wenyang/hcy/TCP_data_collection" # WENYANG NOTE: sensecore version
        dir_root = "/data/wenyang/TCP_data_collection" # WENYANG NOTE: titanxp version
        if self._casetype_town12 is None:
            for dir_name in list(reversed(os.listdir(f'{dir_root}/{town}/'))):
                if dir_name.startswith(f'{town}_scenario{self._data.scenario_tree.name.lower().split("_")[-1]}_'):
                    # data_dir = os.path.join(f'/mnt/afs/user/xiejiangwei/hcy/TCP_data_collection/{town}/', dir_name)
                    data_dir = os.path.join(f'{dir_root}/{town}/', dir_name)
                    break
        else:
            for dir_name in os.listdir(f'{dir_root}/{town}/{self._casetype_town12}/')[::-1]:
                prefix = f'{town}_{self._casetype_town12}_scenario{self._data.scenario_tree.name.lower().split("_")[-1]}_'
                if dir_name.startswith(prefix):
                    # data_dir = os.path.join(f'/mnt/afs/user/xiejiangwei/hcy/TCP_data_collection/{town}/', dir_name)
                    data_dir = os.path.join(f'{dir_root}/{town}/{self._casetype_town12}/', dir_name)
                    self.data_dir = data_dir
                    break
        if data_dir == '':
            print('Save data dir not found')
            exit(0)


        # Create the title
        output = "\n"
        output += "\033[1m========= Results of {} (repetition {}) ------ {} \033[1m=========\033[0m\n".format(
            self._data.scenario_tree.name, self._data.repetition_number, self._global_result)
        output += "\n"
        data['global'] = str(self._global_result)

        # Simulation part
        system_time = round(self._data.scenario_duration_system, 2)
        game_time = round(self._data.scenario_duration_game, 2)
        ratio = round(self._data.scenario_duration_game / self._data.scenario_duration_system, 3)

        list_statistics = [["Start Time", "{}".format(self._start_time)]]
        list_statistics.extend([["End Time", "{}".format(self._end_time)]])
        list_statistics.extend([["Duration (System Time)", "{}s".format(system_time)]])
        list_statistics.extend([["Duration (Game Time)", "{}s".format(game_time)]])
        list_statistics.extend([["Ratio (System Time / Game Time)", "{}".format(ratio)]])

        data['time'] = {}
        data['time']['start_time'] = str(self._start_time)
        data['time']['end_time'] = str(self._end_time)
        data['time']['duration_system'] = system_time
        data['time']['duration_game'] = game_time
        data['time']['ratio'] = ratio

        output += tabulate(list_statistics, tablefmt='fancy_grid')
        output += "\n\n"

        # Criteria part
        header = ['Criterion', 'Result', 'Value']
        list_statistics = [header]
        data['criteria'] = {}
        criteria_data = OrderedDict()

        for criterion in self._data.scenario.get_criteria():

            name = criterion.name

            if name in criteria_data:
                result = criterion.test_status
                if STATUS_PRIORITY[result] < STATUS_PRIORITY[criteria_data[name]['result']]:
                    criteria_data[name]['result'] = result
                criteria_data[name]['actual_value'] += criterion.actual_value

            else:
                criteria_data[name] = {
                    'result': criterion.test_status,
                    'actual_value': criterion.actual_value,
                    'expected_value': criterion.success_value,
                    'units': criterion.units
                }


        for criterion_name in criteria_data:
            criterion = criteria_data[criterion_name]

            result = criterion['result']
            if result in COLORED_STATUS:
                result = COLORED_STATUS[result]

            if criterion['units'] is None:
                actual_value = ""
            else:
                actual_value = str(criterion['actual_value']) + " " + criterion['units']

            data['criteria'][criterion_name] = {
                'result': criteria_data[criterion_name]['result'],
                'actual_value': actual_value,
                'expected_value': criteria_data[criterion_name]['expected_value'],
            }
            list_statistics.extend([[criterion_name, result, actual_value]])

        # Timeout
        name = "Timeout"

        actual_value = self._data.scenario_duration_game

        if self._data.scenario_duration_game < self._data.scenario.timeout:
            result = '\033[92m'+'SUCCESS'+'\033[0m'
            data['criteria']['timeout'] = {'result': False}
        else:
            result = '\033[91m'+'FAILURE'+'\033[0m'
            data['criteria']['timeout'] = {'result': True}

        list_statistics.extend([[name, result, '']])

        output += tabulate(list_statistics, tablefmt='fancy_grid')
        output += "\n"

        key_map = Edict({
            'data_dir': data_dir,
            'images': {
                'rgb1': 'front-left',
                'rgb2': 'front',
                'rgb3': 'front-right',
                'rgb4': 'back-left',
                'rgb5': 'back',
                'rgb6': 'back-right',
                'visualize': 'visualize',
            },
            'more_info': 'more_info',
            'measurements': 'measurements'
        })
        with open(os.path.join(data_dir, 'key_map.json'), 'w') as f:
            json.dump(key_map, f, indent=4, ensure_ascii=False)

        # image2video(key_map) # WENYANG DEBUG 关闭生成视频

        with open(os.path.join(data_dir, 'statistic.json'), 'w') as f:
            json.dump(Edict(data), f, indent=4, ensure_ascii=False)
        print(f'Statistic saved to {os.path.join(data_dir, "statistic.json")}')

        return output
    
    

