# bachelor_thesis
Program of my bachelor thesis, Jad Dayoub 7425569.

The goal is to improve pulsar detection by using customized filters for CNNs. With enhanced filter initialization, the CNNs are expected to converge faster and achieve greater accuracy.

The Deep Learning Framework PyTorch is used to create and train the networks. Therefore to be able to execute the programm this framework is necessary (see https://pytorch.org/get-started/locally/ for further information).

I've got provided access to the Machine Learning Pipeline for Pulsar Analysis from PUNCH4NFDI, to generate synthetic data. The DM values are taken from https://www.atnf.csiro.au/people/pulsar/psrcat/ and a frequency range of 1.21 - 1.53GHz is used.

Example:

![grafik](https://github.com/undefined-panda/bachelor_thesis/assets/154523220/d934c68c-4bda-4239-b363-d5b93b544cb8)

The filters I've used have their usage in image processing:
- Prewitt
- Sobel
- Kirsch
- Canny-Algorithm

To see the results for the different filters, please refer to `results`.

To read the thesis, please refer to `Bachelorarbeit_Jad_Dayoub.pdf` (currently only in german).