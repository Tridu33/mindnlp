#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
"""
python utils.py
"""


def rename_en2zh_for_particular_file(top):
    for root, dirs, files in os.walk(top, topdown=False):
        for file_name in files:
            if len(file_name.split('.'))==3:
                if file_name.split('.')[-1] == 'md' and file_name.split('.')[-2] == 'zh':
                    to_be_renamed_file_name = os.path.join(root, file_name)
                    new_name = to_be_renamed_file_name.split('.')[0] + '.md'
                    os.rename(to_be_renamed_file_name,new_name)


def delete_particular_file(top):
    for root, dirs, files in os.walk(top, topdown=False):
        for file_name in files:
            if file_name.split('.')[-1] == 'vswp':
                delete_file_name = os.path.join(root, file_name)
                os.remove(delete_file_name)
                print(f'{delete_file_name} deleted...')
            if file_name.split('.')[-1] == 'json':
                if file_name.split('.')[0] == '_vnote' or file_name.split('.')[0] == 'vx':
                    delete_file_name = os.path.join(root, file_name)
                    os.remove(delete_file_name)
                    print(f'{delete_file_name} deleted...')
            # for name in dirs:
            #     os.rmdir(os.path.join(root, name))


def print_config_yaml_files_names_under_en_folder(top):
    for root, dirs, files in os.walk(top, topdown=False):
        for file_name in files:
            if len(file_name.split('.'))==2:
                if file_name.split('.')[-1] == 'md':
                    print("  - '"+file_name.split('.')[0]+"'")

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # print_config_yaml_files_names_under_en_folder(current_dir)
    rename_en2zh_for_particular_file(current_dir)
