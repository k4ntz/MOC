FROM nvcr.io/nvidia/pytorch:21.03-py3

RUN     git clone https://github.com/ml-research/SPACE.git
CMD     cd SPACE

