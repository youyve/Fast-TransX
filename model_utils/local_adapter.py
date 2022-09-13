"""Local adapter"""

import os


def get_device_id():
    """Get device ID"""
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)


def get_device_num():
    """Get number of available devices"""
    device_num = os.getenv('RANK_SIZE', '1')
    return int(device_num)


def get_rank_id():
    """Get the rank of the process"""
    global_rank_id = os.getenv('RANK_ID', '0')
    return int(global_rank_id)


def get_job_id():
    """Get a job"""
    return "Local Job"
