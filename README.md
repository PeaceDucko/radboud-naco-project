# Image Classification with Puissance-assisted Neuroevolution

"Image Classification with Puissance-assisted Neuroevolution" is a Natural computing project made by students of Radboud University.  
This project applies Mutational Puissance Assisted Neuroevolution to NEAT. 


## Installation
The code on this repository was tested on Python 3.6.9.

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the dependencies for this project from the root folder.

```bash
pip install -r requirements.txt
```

Optionally, you could create a virtual environment using the Python virtualenv package by running
```bash
virtualenv <env_name>
source <env_name>/bin/activate
pip install -r requirements.txt
```

## Usage

There are 2 ways this project can be ran:
* Run Sklearn-neuroevolution on the CIFAR-10 dataset.
* Run Sklearn-neuroevolution with Mutational Puissance on the CIFAR-10 dataset.

To run the normal Sklearn-neuroevolution on the CIFAR-10 dataset.
```bash
mv ./Sklearn-neat/neat /Sklearn-neat/temp-neat
python ./Sklearn-neat/main.py
```
changing the directory name for neat will ensure that Sklearn-neuroevolution will import neat normally instead of using the locally modified version that should be used in th example below.

To run Sklearn-neuroevolution with Mutational Puissance on the CIFAR-10 dataset.

```bash
python ./Sklearn-neat/main.py
```

## Contributors
<table>
  <tr>
    <td align="center"><a href="https://github.com/PeaceDucko"><img src="https://avatars.githubusercontent.com/u/36194484?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Sebastiaan Ram</b>
    </td>
    <td align="center"><a href="https://github.com/Ylja07"><img src="https://avatars.githubusercontent.com/u/27802933?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Ylja van Son</b>
    </td>
    <td align="center"><a href="https://github.com/W-M-T"><img src="https://avatars.githubusercontent.com/u/12382856?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Ward Theunisse</b>
    </td>
  </tr>
</table>
