import os
import datetime


def check_and_create_tmp_dir(tmp_dir):
    if not os.path.exists(tmp_dir):
        print(f"WARNING: {tmp_dir} does not exists")
        print(f"Creating {tmp_dir}")
        os.makedirs(tmp_dir)


def iso_time():
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")
