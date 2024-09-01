import time

from src.util.ssh import SSHConnection

Slaves = ['xxx.xxx.xxx.xxx', 'xxx.xxx.xxx.xxx', 'xxx.xxx.xxx.xxx']
Master = ['xxx.xxx.xxx.xxx']
UserName = 'anonymous'

class MemoryMonitor:
    def __init__(self, bench_time, bench_interval):
        self.TaskManagers = []
        self.JobManager = None
        self.TaskManagerPIDs = []
        self.JobManagerPID = None

        self.bench_time = bench_time
        self.bench_interval = bench_interval

        for slave in Slaves:
            ssh = SSHConnection(hostname=slave, username=UserName)
            ssh.connect()
            stdin, stdout, stderr = ssh.cmd('jps | grep TaskManager')
            self.TaskManagerPIDs.append(str(stdout.readline()).split(' ')[0])
            self.TaskManagers.append(ssh)

        ssh = SSHConnection(hostname=Master[0], username=UserName)
        ssh.connect()
        self.JobManager = ssh
        stdin, stdout, stderr = self.JobManager.cmd('jps | grep StandaloneSessionClusterEntrypoint')
        self.JobManagerPID = str(stdout.readline()).split(' ')[0]

    def start(self):
        average_memory_usage = []

        sleep_times = self.bench_time / self.bench_interval

        j = 0
        while j < sleep_times:
            sum_memory = 0

            # JobManager
            stdin, stdout, stderr = self.TaskManagers[0].cmd(
                'cat /proc/{}/status | grep VmRSS'.format(self.JobManagerPID))
            JM_res = stdout.readline()
            for s in JM_res.split(' '):
                if s.isdigit():
                    try:
                        sum_memory += int(s)
                    except Exception as e:
                        print('maybe Flink process is not running')
                        sum_memory += 0

            # TaskManagers
            for i in range(len(self.TaskManagerPIDs)):
                stdin, stdout, stderr = self.TaskManagers[i].cmd(
                    'cat /proc/{}/status | grep VmRSS'.format(self.TaskManagerPIDs[i]))
                TM_res = stdout.readline()
                print('TM res:', TM_res)

                r = ''

                for char in TM_res:
                    if char.isdigit():
                        r = r + char

                try:
                    sum_memory += int(r)
                    print('int(r)', int(r))
                except Exception as e:
                    print('maybe Flink process is not running')
                    sum_memory += 0
            average_memory_usage.append(sum_memory)
            print('Memory', sum_memory)
            time.sleep(self.bench_interval)
            j += 1

        print(average_memory_usage)
        # finished
        for SSH in self.TaskManagers:
            SSH.close()

        return sum(average_memory_usage) / len(average_memory_usage)


if __name__ == "__main__":
    st = time.time()
    m = MemoryMonitor(170, 1)
    print(m.start())
    print(time.time() - st)
