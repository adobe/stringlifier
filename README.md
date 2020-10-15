[![Downloads](https://pepy.tech/badge/stringlifier)](https://pepy.tech/project/stringlifier) [![Downloads](https://pepy.tech/badge/stringlifier/month)](https://pepy.tech/project/stringlifier/month) ![Weekly](https://img.shields.io/pypi/dw/stringlifier.svg) ![daily](https://img.shields.io/pypi/dd/stringlifier.svg)
![Version](https://badge.fury.io/py/stringlifier.svg) [![Python 3](https://img.shields.io/badge/python-3-blue.svg)](https://www.python.org/downloads/release/python-360/) [![GitHub stars](https://img.shields.io/github/stars/adobe/stringlifier.svg?style=social&label=Star&maxAge=2592000)](https://github.com/adobe/stringlifier/stargazers/)

# stringlifier
String-classifier - is a python module for detecting random string and hashes text/code. 

Typical usage scenarios include:

* Sanitizing application or security logs
* Detecting accidentally exposed credentials (complex passwords or api keys)

# Interactive notebook

You can see Stringlifier in action by checking out this [interactive notebook hosted on Colaboratory](https://colab.research.google.com/drive/1bgZQSKhVAYU4r46wqb0v8Sfvuo_yMOLA?usp=sharing).

# Quick start guide

You can quickly use stringlifier via pip-installation:
```bash
$ pip install stringlifier
```
In case you are using the pip3 installation that comes with Python3, use pip3 instead of pip in the above command.
```bash
$ pip3 install stringlifier
```

API example:
```python
from stringlifier.api import Stringlifier

stringlifier=Stringlifier()

s = stringlifier("com.docker.hyperkit -A -u -F vms/0/hyperkit.pid -c 8 -m 8192M -b 127.0.0.1 --pass=\"NlcXVpYWRvcg\" -s 0:0,hostbridge -s 31,lpc -s 1:0,virtio-vpnkit,path=vpnkit.eth.sock,uuid=45172425-08d1-41ec-9d13-437481803412 -U c6fb5010-a83e-4f74-9a5a-50d9086b9")
```

After this, `s` should be:

```'com.docker.hyperkit -A -u -F vms/0/hyperkit.pid -c 8 -m 8192M -b <IP_ADDR> --pass="<RANDOM_STRING>" -s 0:0,hostbridge -s 31,lpc -s 1:0,virtio-vpnkit,path=vpnkit.eth.sock,uuid=<UUID> -U <UUID>'```

You can also choose to see the full tokenization and classification output:

```python
s, tokens = stringlifier("com.docker.hyperkit -A -u -F vms/0/hyperkit.pid -c 8 -m 8192M -b 127.0.0.1 --pass=\"NlcXVpYWRvcg\" -s 0:0,hostbridge -s 31,lpc -s 1:0,virtio-vpnkit,path=vpnkit.eth.sock,uuid=45172425-08d1-41ec-9d13-437481803412 -U c6fb5010-a83e-4f74-9a5a-50d9086b9", return_tokens=True)
```

`s` will be the same as before and `tokens` will contain the following data:
```python
[[('0', 33, 34, '<NUMERIC>'),
   ('8', 51, 52, '<NUMERIC>'),
   ('8192', 56, 60, '<NUMERIC>'),
   ('127.0.0.1', 65, 74, '<IP_ADDR>'),
   ('NlcXVpYWRvcg', 83, 95, '<RANDOM_STRING>'),
   ('0', 100, 101, '<NUMERIC>'),
   ('0', 102, 103, '<NUMERIC>'),
   ('31', 118, 120, '<NUMERIC>'),
   ('1', 128, 129, '<NUMERIC>'),
   ('0', 130, 131, '<NUMERIC>'),
   ('45172425-08d1-41ec-9d13-437481803412', 172, 208, '<UUID>'),
   ('c6fb5010-a83e-4f74-9a5a-50d9086b9', 212, 244, '<UUID>')]]
```



# Building your own classifier

You can also train your own model if you want to detect different types of strings. For this you can use the Command Line Interface for the string classifier:

```bash
$ python3 stringlifier/modules/stringc.py --help

Usage: stringc.py [options]

Options:
  -h, --help            show this help message and exit
  --interactive
  --train
  --resume
  --train-file=TRAIN_FILE
  --dev-file=DEV_FILE
  --store=OUTPUT_BASE
  --patience=PATIENCE   (default=20)
  --batch-size=BATCH_SIZE
                        (default=32)
  --device=DEVICE
```

For instructions on how to generate your training data, use [this link](corpus/README.md).

**Important note:** This model might not scale if detecting a type of string depends on the surrounding tokens. In this case, you can look at a more advanced tool for sequence processing such as [NLP-Cube](https://github.com/adobe/NLP-Cube)
