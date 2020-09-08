from time import strftime
import os
import socket
from torch.utils.tensorboard import SummaryWriter
from subprocess import Popen
from mtrain.recorder.info_print import cprint, WARMN, HINTS

def killport(port):
    command = "fuser -k {0}/tcp".format(port)
    print(command)
    os.system(command)

def tryPort(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = False
    try:
        sock.bind(("0.0.0.0", port))
        result = True
    except:
        print("Port is in use")
    sock.close()
    return result

def build_dir(model_name):
    time_stamp = strftime("[%m-%d] %H-%M-%S")
    log_path = os.path.join("records", model_name, time_stamp)
    return log_path


class TBHandler(object):


    def __init__(self, name):
        self.log_dir = build_dir(name)
        self.model_name = name
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def get_writer(self):
        return self.writer

    def star_shell_tb(self, port):
        killport(port)
        cmd = 'tensorboard --logdir "{l}" --reload_interval 5 --port {p}'.format(l=self.log_dir, p=port)
        tb_poc = Popen(cmd, shell=True)
        self.cmd = cmd
        self.tb_poc = tb_poc
        cprint(WARMN, "Tensorboard running at http:://127.0.0.1:{p}".format(p=port))


    def kill_shell_tb(self):
        self.tb_poc.kill()
        hint_return = 'Using "{}" to cheack training process.'.format(self.cmd)
        cprint(HINTS, hint_return)



