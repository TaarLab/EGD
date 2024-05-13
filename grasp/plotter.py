import time
from janus import Queue
from threading import Thread


class PlotterThread(Thread):
    def __init__(self):
        super().__init__()
        self.commands = Queue().sync_q
        self.interval = None

    def run(self):
        while True:
            if self.commands:
                while self.commands.qsize() > 0:
                    command = self.commands.get()
                    command()
            if self.interval:
                self.interval()
            time.sleep(0.01)

    def add_command(self, cmd):
        self.commands.put(cmd)

    def set_interval(self, cmd):
        self.interval = cmd