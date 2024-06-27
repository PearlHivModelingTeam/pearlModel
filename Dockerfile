FROM condaforge/mambaforge:4.9.2-5 as conda

# Add requirements.txt
ADD requirements.txt /tmp/requirements.txt

#upgrade pip
RUN python -m pip install --upgrade pip

# create a conda env
ENV CONDA_ENV ./conda/bin
RUN conda create -n "myenv" python=3.11.9
ENV PATH "${CONDA_ENV}_ENV}/bin:${PATH}"

# install all pip requirements
RUN conda install --yes --file /tmp/requirements.txt