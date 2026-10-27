"""Run a command on the compute server inside ~/research/grag.
usage: python _remote.py "<command>"            run a shell command
       python _remote.py --put local remote      upload a file
       python _remote.py --get remote local      download a file"""
import os  # public release: SSH configuration comes from the environment
import sys, paramiko, warnings
warnings.filterwarnings("ignore")
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(os.environ["SSH_HOST"], int(os.environ.get("SSH_PORT", "22")), os.environ["SSH_USER"], os.environ.get("SSH_PASSWORD"), timeout=30)
H = "/home/yaokang/research/grag/"
if sys.argv[1] == "--put":
    c.open_sftp().put(sys.argv[2], H + sys.argv[3])
elif sys.argv[1] == "--get":
    r = sys.argv[2]
    c.open_sftp().get(r if r.startswith("/") else H + r, sys.argv[3])
else:
    _, o, e = c.exec_command("cd ~/research/grag; " + sys.argv[1], timeout=900)
    sys.stdout.buffer.write(o.read()); sys.stdout.buffer.write(e.read()[-3000:])
