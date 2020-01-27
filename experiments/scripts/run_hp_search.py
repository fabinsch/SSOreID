import os

#os.system("/opt/anaconda3/bin/python -u experiments/scripts/hp_search_finetune_tracktor.py with cfg1")
#os.system("/home/carolin/tracking_wo_bnw/torch13/bin/python -u experiments/scripts/hp_search_finetune_tracktor.py with cfg1")
os.system("cd ../.. && /home/carolin/tracking_wo_bnw/torch13/bin/python  -u experiments/scripts/finetune_independently.py with cfg1")
