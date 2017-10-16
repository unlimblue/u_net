# -*- coding: utf-8 -*-


"""
MIT License

Copyright (c) 2017 sli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import traceback

from queue import Queue as th_queue
from threading import Thread

class thread_worker(Thread):
    """
    从队列tasks中获取任务，交由所持有的线程执行
    """
    def __init__(self, tasks):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.start()
    
    def run(self):
        while True:
            func, args, kwargs = self.tasks.get()
            try:
                func(*args, **kwargs)
            except Exception:
                traceback.print_exc()
            self.tasks.task_done()

class thread_pool:
    """
    轻量级线程池
    """
    def __init__(self, num_threads, max_tasks=None):
        if not max_tasks:
            self.tasks = th_queue()
        else:
            self.tasks = th_queue(max_tasks)
        self.workers = [thread_worker(self.tasks) for _ in range(num_threads)]

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.wait_completion()

    def add_task(self, func, *args, **kwargs):
        """
        向线程池中增加任务

        参数
            func 任务函数
            *args, **kwargs 任务函数参数
        """
        self.tasks.put((func, args, kwargs))

    def wait_completion(self):
        """
        等待线程完成后，退出
        """
        self.tasks.join()

