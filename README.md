# stringlifier
String-classifier - is a python module for detecting random string and hashes text/code. 

Typical usage scenarios include:

* Sanitizing application or security logs
* Detecting accidentally exposed credentials (complex passwords or api keys)


# Quick start guide

You can quickly use stringlifier via pip-installation:
```bash
$ pip install stringlifier
```

API example:
```python
from stringlifier.api import Stringlifier

stringlifier=Stringlifier()

s = stringlifier('/System/Library/DriverExtensions/AppleUserHIDDrivers.dext/AppleUserHIDDrivers com.apple.driverkit.AppleUserUSBHostHIDDevice0 0x10000992d')
```

After this, `s` should be:

```'/System/Library/DriverExtensions/AppleUserHIDDrivers.dext/AppleUserHIDDrivers com.apple.driverkit.AppleUserUSBHostHIDDevice0 <RANDOM_STRING>'```

You can also choose to see the full tokenization and classification output:

```python
s, tokens = stringlifier('/System/Library/DriverExtensions/AppleUserHIDDrivers.dext/AppleUserHIDDrivers com.apple.driverkit.AppleUserUSBHostHIDDevice0 0x10000992d', return_tokens=True)
```

`s` will be the same as before and tokens will contain the following data:
```python
[{'token': '/', 'type': 'SYMBOL'},
 {'token': 'System', 'type': 'STRING'},
 {'token': '/', 'type': 'SYMBOL'},
 {'token': 'Library', 'type': 'STRING'},
 {'token': '/', 'type': 'SYMBOL'},
 {'token': 'DriverExtensions', 'type': 'STRING'},
 {'token': '/', 'type': 'SYMBOL'},
 {'token': 'AppleUserHIDDrivers', 'type': 'STRING'},
 {'token': '.', 'type': 'SYMBOL'},
 {'token': 'dext', 'type': 'STRING'},
 {'token': '/', 'type': 'SYMBOL'},
 {'token': 'AppleUserHIDDrivers', 'type': 'STRING'},
 {'token': ' ', 'type': 'SYMBOL'},
 {'token': 'com', 'type': 'STRING'},
 {'token': '.', 'type': 'SYMBOL'},
 {'token': 'apple', 'type': 'STRING'},
 {'token': '.', 'type': 'SYMBOL'},
 {'token': 'driverkit', 'type': 'STRING'},
 {'token': '.', 'type': 'SYMBOL'},
 {'token': 'AppleUserUSBHostHIDDevice0', 'type': 'STRING'},
 {'token': ' ', 'type': 'SYMBOL'},
 {'token': '0x10000992d', 'type': 'HASH'}]
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
