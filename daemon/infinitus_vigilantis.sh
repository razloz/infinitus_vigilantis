#!/usr/bin/bash
cd $IVY && $IVY/updater.py --build && $IVY/updater.py --validate --start_date 2020-01-01 && $IVY/cartographer.py --all --size 160 --start_date 2020-01-01
