# dev_scripts/high_freq_all.sh

import subprocess
from datetime import datetime
import pytz
import time

# 获取当前北京时间
def get_beijing_time():
    beijing_tz = pytz.timezone('Asia/Shanghai')
    beijing_time = datetime.now(beijing_tz)
    return beijing_time.strftime("%Y-%m-%d %H:%M:%S")

# 获取 nvidia-smi 输出并解析空闲显存
def get_gpu_free_memory():
    # 执行 nvidia-smi 命令获取 GPU 的显存信息
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8').strip().split('\n')
    
    gpu_free_memory = []
    for line in output:
        index, free_memory = line.split(', ')
        gpu_free_memory.append((int(index), int(free_memory)))
    
    return gpu_free_memory

# 检查空闲显存并执行 bash 脚本
def check_and_run_script():
    while True:  # 持续循环，直到手动停止
        gpu_free_memory = get_gpu_free_memory()
        for index, free_memory in gpu_free_memory:
            if free_memory > 22000:  # 检查空闲显存是否大于 22GB
                beijing_time = get_beijing_time()
                print(f"GPU {index} has {free_memory}MB free memory. (Time: {beijing_time})")
                print(f"GPU {index} has more than 22GB free memory. Running run.sh on GPU {index}... (Time: {beijing_time})")
                
                # 执行 bash 脚本并等待其完成
                subprocess.run(['bash', ' dev_scripts/high_freq_all.sh', str(index)])
                print(f"run.sh executed for GPU {index}. Waiting for next check...")
                break
        
        # 每次检查后等待一定时间再继续检查
        time.sleep(5)  # 每隔 5 秒检查一次 GPU 空闲显存

# 执行检测
check_and_run_script()
