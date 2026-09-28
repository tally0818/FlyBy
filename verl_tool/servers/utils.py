import os
import signal
import subprocess
import sys
import hashlib

def kill_python_subprocess_processes():
    'Kill any lingering Python processes that were spawned with the -c flag.'
    try:

        ps_process = subprocess.Popen(
            ["ps", "-ef"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, _ = ps_process.communicate()
        

        own_pid = os.getpid()
        ps_pid = ps_process.pid
        
        killed_count = 0
        
        for line in stdout.splitlines():
            parts = line.split()
            if len(parts) < 8:
                continue
                
            pid_str = parts[1]

            cmd = " ".join(parts[7:])
            

            if (("python -c" in cmd or "python3 -c" in cmd) and 
                pid_str.isdigit()):
                pid = int(pid_str)
                

                if pid != own_pid and pid != ps_pid:
                    try:

                        os.kill(pid, signal.SIGKILL)
                        killed_count += 1
                    except (ProcessLookupError, PermissionError) as e:

                        print(f"Error killing process {pid}: {e}")
        
        return killed_count
            
    except Exception as e:
        print(f"Error during process cleanup: {e}")
        return 0
    
    
def hash_requests(data):
    'Hash the input data to create a unique identifier.'

    data_str = str(data).encode('utf-8')
    hash_object = hashlib.sha256()
    hash_object.update(data_str)
    return hash_object.hexdigest()