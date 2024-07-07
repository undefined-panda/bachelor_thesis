# bachelor_thesis
Program of my bachelor thesis 2024.

The objective is to improved pulsar detection by creating customized filters for CNNs. With enhanced filter initialization, the CNNs are expected to converge faster and achieve greater accuracy.

The Deep Learning Framework PyTorch is used to create and train the networks. Therefore to be able to execute the programm this framework is necessary (see https://pytorch.org/get-started/locally/ for further information).

Part of my thesis was to create my own synthesized dataset. The code can be found in ```synthetic_data_generation.py```. An exponential decaying function is employed to simulate the shape of a pulsar image, replicating its dispersion delay frequency measurement.

Examples:

![grafik](https://github.com/undefined-panda/bachelor_thesis/assets/154523220/0590e7e6-384e-4054-ab71-01b6ece1e55c)
![grafik](https://github.com/undefined-panda/bachelor_thesis/assets/154523220/20e592b7-b812-42c9-bc8c-d5f966b39e19)
![grafik](https://github.com/undefined-panda/bachelor_thesis/assets/154523220/89383b8a-23c5-42ae-aa39-0075d816a666)

I've got also provided access to the Machine Learning Pipeline for Pulsar Analysis from PUNCH4NFDI, to generate synthetic data. The DM values are taken from https://www.atnf.csiro.au/people/pulsar/psrcat/ and a frequency range of 1.21 - 1.53GHz is used.

Example:

![grafik](https://github.com/undefined-panda/bachelor_thesis/assets/154523220/d934c68c-4bda-4239-b363-d5b93b544cb8)

To execute the tests run ```main.ipynb```.

- 32x32 works with a learning rate of 0.001