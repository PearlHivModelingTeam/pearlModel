FROM condaforge/mambaforge:4.9.2-5 as conda

# update packages and install make
RUN apt-get update && apt-get install make

# Add environment lock file
ADD environment.yml /tmp/environment.yml

# create a conda env
ENV CONDA_ENV ./conda/bin
RUN conda env create -f /tmp/environment.yml

RUN echo "source activate myenv" > ~/.bashrc
ENV PATH /opt/conda/envs/myenv/bin:$PATH

ADD . /
RUN pip install .
