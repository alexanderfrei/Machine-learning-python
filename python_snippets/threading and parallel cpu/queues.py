
""" Queue and threading """

from queue import Queue
from threading import Thread

job_result = []


def worker(q):
    while True:  # infinity loop cause worker is a daemon
        data = q.get()
        job_result.append(data[1] / 2 )
        print('Task {} done, queue size = {}'.format(data[0], q.qsize()))
        # q.task_done()

queue = Queue(maxsize=2)
num_threads = 4

for i in range(num_threads):
    thread = Thread(target=worker, args=(queue, ))
    thread.setDaemon(True)  # threads killed when program stops
    thread.start()

for x in range(10):
    print('Start task {}'.format(x))
    queue.put((x, x * 10))

# queue.join()  # block until all tasks are done
print(job_result)

