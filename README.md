# EEG Motor Imagery Classification in Node.js with BCI.js Tutorial Code

Here you can find the code and data which accompany the [BCI.js](https://github.com/pwstegman/bcijs) motor imagery tutorial.

## Getting Started

From within this project's directory, run

```bash
npm install
```

Then for classification results which allow unknown values, run

```bash
node ./classifyImagery-unknowns.js
```

For classification results which don't allow unknown values, run

```bash
node ./classifyImagery-noUnknowns.js
```

## Data

[A01T.mat](data/A01T.mat) and [A01E.mat](data/A01E.mat) contain the training and evaluation data respectively. They were converted to a CSV format using the Octave/MATLAB script [SubjectDataToCSV.m](SubjectDataToCSV.m).

Data files were downloaded from [http://bnci-horizon-2020.eu/database/data-sets](http://bnci-horizon-2020.eu/database/data-sets)<br />
License: [Creative Commons Attribution No Derivatives license (CC BY-ND 4.0)](https://creativecommons.org/licenses/by-nd/4.0/).<br />
Licensor: [Institute for Knowledge Discovery, Graz University of Technology](http://bci.tugraz.at/)
