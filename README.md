# radboud-naco-project

radboud naco project is a Natural computing project made by students of Rabdoub University.  
This project applies Mutational Puissance Assisted Neuroevolution to NEAT. 


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the dependencies for this project.

```bash
pip install -r requirements.txt
```

## Usage

There are 2 ways this project can be ran:
* Run Sklearn-neuroevolution on the CIFAR-10 dataset.
* Run Sklearn-neuroevolution with Mutational Puissance on the CIFAR-10 dataset.

To run the normal Sklearn-neuroevolution on the CIFAR-10 dataset.
```bash
mv /Sklearn-neat/neat /Sklearn-neat/temp-neat
py /Sklearn-neat/main.py
```
Changing the directory name for neat will ensure that Sklearn-neuroevolution will import neat normally instead of using the locally modified version that should be used in the example below.

<br>
To run Sklearn-neuroevolution with Mutational Puissance on the CIFAR-10 dataset.

```bash
py /Sklearn-neat/main.py
```

## Contribors
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
