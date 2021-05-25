#!/usr/bin/bash
cd $IVY && $IVY/updater.py --build && $IVY/updater.py --validate --start_date 2021-01-01 && $IVY/cartographer.py --all --size 320 --start_date 2021-01-01
