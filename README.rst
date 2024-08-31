===============
The PEARL Model
===============

The ProjEcting Age, multimoRbidity, and poLypharmacy (PEARL) model is an agent-based simulation 
model of persons living with HIV (PWH) using antiretroviral therapy (ART) in the US (2009 - 2030). 
Due to effective treatment, PWH accessing care in the US now have a life expectancy approaching the 
general population. As these people survive to older ages, the burden of multimorbidity and 
polypharmacy will change to reflect this older age distribution and non-HIV-related healthcare 
costs will become increasingly important in caring for these individuals. Since the relevant 
results vary greatly by demographic among PWH in the US, the PEARL model accounts race, sex, HIV 
risk group explicitly. Among these key populations, outcomes depend further upon age, age at ART 
initiation, CD4 count at ART initiation, current CD4 count, and BMI. For most of its machinery, the 
PEARL model utilizes data from th North American AIDS Cohort Collaboration on Research and Design 
(NA-ACCORD). The NA-ACCORD is comprised of data from over 20 cohorts, 200 sites, and 190,000 
HIV-infected participants, providing a data set that is widely representative of HIV care in 
North America. Data on overall PWH population size comes mostly from CDC surveillance data.

The PEARL model has been constructed to achieve the following:

**Aim 1:** To fill the gap in knowledge by projecting age distributions of PWH using ART in the US 
through 2030 broken down by key population.

**Aim 2:** To project the burden of multimorbidity and polypharmacy among PWH using ART in the US 
through 2030.

**Aim 3:** To project the annual costs of non-HIV-related healthcare for PWH using ART in the US 
through 2030.

==========================
Installation and First Run
==========================

Clone the repository onto your machine, enter the directory and install pearl::

    git clone git@github.com:PearlHivModelingTeam/pearlModel.git
    cd pearlModel
    pip install .

The ``scripts`` folder holds 3 numbered python scripts as well as a library with the PEARL classes 
and variables. The python files are numbered so that they can be run one after another to run a 
simulation. The ``config`` folder holds yaml files for specifying run configurations, the 
``param_files`` folder holds the input parameters for use by PEARL and simulation results are 
generated in the ``out`` folder.

Finally, enter the ``scripts`` folder and run the first two numbered scripts. This will generate 
parameters and run a simulation using the test.yaml config file. The simulation output can be found 
in ``out/test_yyyy_mm_dd`` with the date corresponding to the initiation of the run::


    python scripts/1_create_param_file.py
    python scripts/2_simulate.py
    python scripts/3_combine_parquet.py --in_dir path/to/out/dir/parquet_output


=======================
Development Environment
=======================

For development, and usage, we suggest using docker and vscode, with instructions outlined below:

^^^^^^^^^^^^^^^^^^^^^^
Step 1: Install VSCode
^^^^^^^^^^^^^^^^^^^^^^
1. Navigate to the `Visual Studio Code website <https://code.visualstudio.com/>`_.
2. Download the appropriate installer for your operating system (Windows, Linux, or macOS).
3. Run the installer and follow the on-screen instructions to install VSCode on your system.
4. After installation, launch VSCode.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 2: Install DevContainer Extension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. In VSCode, go to the Extensions view by clicking on the Extensions icon in the Activity Bar on 
the side of the window.
2. Search for "Dev Containers" in the Extensions view search bar.
3. Find the "Dev Containers" extension in the search results and click on the install button to 
install it.

You can also go to the extension's 
`homepage <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers>`_ 
and 
`documentation page <https://code.visualstudio.com/docs/devcontainers/containers>`_ 
to find more details.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 3: Install Docker and Add Current Login User to Docker Group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Follow the `official guide <https://docs.docker.com/get-docker/>`_ to install Docker. Don't forget 
the `post installation steps <https://docs.docker.com/engine/install/linux-postinstall/>`_.

If you are using `Visual Studio Code Remote - SSH <https://code.visualstudio.com/docs/remote/ssh>`_, 
then you only need to install Docker in the remote host, not your local computer. And the following steps should be run in the remote host.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 4: Open in DevContainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In VSCode, use the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P` on macOS) to run the 
"Dev Containers: Open Folder in Container..." command.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 5: Wait for Building the Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. After opening the folder in a DevContainer, VSCode will start building the container. This 
process can take some time as it involves downloading necessary images and setting up the 
environment.

2. You can monitor the progress in the VSCode terminal.

3. Once the build process completes, you'll have a fully configured development environment in a 
container.

4. The next time you open the same dev container, it will be much faster, as it does not require 
building the image again.


-------
Testing
-------
To ensure that the package is working as intended, can run the test suit with:

``pytest tests``

=============
Configuration
=============

The simulation script can be called with a command line argument pointing to a config file in order 
to run a simulation with different parameters or attributes. A template file lives 
``config/template.yaml`` which contains all of the options available. 
In order to run a simulation with a specific config file simply call the simulation script as:
```
python 2_simulate --config my_config.yaml
```
and the output will be created at ``out/my_config_yyyy_mm_dd``

^^^^^^^^^^^^^^^
``group_names``
^^^^^^^^^^^^^^^
A list of the sex, race, and hiv-acquisition groups to include in the simulation. 
Can be any number of 
```
['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'idu_hisp_female', 'het_white_male',
'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']
```

^^^^^^^^^^^^^^^^
``replications``
^^^^^^^^^^^^^^^^
Number of replications of each simulation to run with different seeds. Any positive integer.

^^^^^^^^^^
``new_dx``
^^^^^^^^^^
String indicating which set of parameters to use for number of new diagnoses. 
``base``, ``ehe``, ``sa``. 
The alternate models correspond to models used in some previous papers.

^^^^^^^^^^^^^^^^^^^
``mortality_model``
^^^^^^^^^^^^^^^^^^^
String corresponding to which model to use for mortality. 
``by_sex_race_risk``, ``by_sex_race``, ``by_sex``, ``overall``. 
These models are presented in the mortality paper.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``mortality_threshold_flag``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Flag indicating whether simulation should include the mortality threshold functionality. 0 or 1.

^^^^^^^^^^^^^^
``final_year``
^^^^^^^^^^^^^^
Year to end the simulation. Integer between 2010 and 2035.

^^^^^^^^^^^^^^^
``sa_variable``
^^^^^^^^^^^^^^^
Supports all comorbidities

^^^^^^^^^^^^^^^^^
``idu_threshold``
^^^^^^^^^^^^^^^^^
String corresponding to the different multipliers available for setting the mortality threshold 
for the idu population above other risk groups. ``2x``, ``5x``, ``10x``.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``bmi_intervention_scenario``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
BMI scenario to run from ``0``, ``1``, ``2``, or ``3``

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``bmi_intervention_start_year``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Year to begin BMI intervention in simulation

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``bmi_intervention_end_year``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Year to end BMI intervention in simulation

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``bmi_intervention_coverage``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Probability of an eligible agent receiving an intervention

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``bmi_intervention_effectiveness``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Efficacy of intervention received by agents receiving intervention.
