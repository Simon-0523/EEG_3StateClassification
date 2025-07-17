import sys

class Logger:
    def __init__(self, filename="output.txt"):
        self.terminal = sys.stdout  # 保存终端标准输出
        self.log = open(filename, "w")  # 打开日志文件

    def write(self, message):
        self.terminal.write(message)  # 输出到终端
        self.log.write(message)       # 输出到文件

    def flush(self):
        self.terminal.flush()  # 确保输出流刷新
        self.log.flush()
    def reopen(self):
        self.log = open(self.filename, "a")  # 以追加模式重新打开文件

# sys.stdout = Logger("output.txt")

# sys.stdout.log.close()