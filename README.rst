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

    git clone https://github.com/PearlHivModelingTeam/pearlModel.git
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
python scripts/2_simulate.py --config config/my_config.yaml
```
and the output will be created at ``out/my_config``, named after the stem of the config file. If
that folder already exists the script raises ``FileExistsError``; pass ``--overwrite`` to delete
and regenerate it.

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

^^^^^^^^^^^^^^^^
``sa_variables``
^^^^^^^^^^^^^^^^
List of parameters to perturb for the sensitivity analysis. An empty list (``[]``) means no
sensitivity analysis is performed. The BMI paper sensitivity analysis uses::

    sa_variables: ['dm_prevalence', 'dm_prevalence_prev', 'dm_incidence', 'pre_art_bmi',
                   'post_art_bmi', 'art_initiators']

================================================
Running the Full Analysis with the ``Snakefile``
================================================

The ``Snakefile`` in the project root wires the numbered scripts together into a single
`Snakemake <https://snakemake.readthedocs.io/>`_ workflow, so that one command generates the
parameter file, runs every simulation scenario, combines and aggregates the parquet output, and
produces the BMI paper figures and the sensitivity analysis tornado plots. Snakemake is already
installed in the dev container (see ``environment.yml``); outside of it, install it with
``conda install -c bioconda snakemake`` or ``pip install snakemake``.

The rules chain together as:

    ``create_params`` → ``simulate`` → ``combine`` / ``aggregate`` / ``aggregate_bmi_cat``
    → ``bmi_paper_outputs`` + ``bmi_SA`` → ``all``

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 1: Create the config files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``simulate`` rule is wildcarded on the config name: for a target ``out/{config}/...`` it runs
``python scripts/2_simulate.py --config config/{config}.yaml --overwrite``. The workflow therefore
requires that a config file exists in ``config/`` for every scenario it needs.

Which scenarios it needs is determined by ``num_replications`` at the top of the ``Snakefile``::

    num_replications = 10

With ``num_replications = 10``, the workflow expects these four config files to exist:

+--------------------------+-------------------------------+----------------------------------------+
| Config file              | ``bmi_intervention_scenario`` | ``sa_variables``                       |
+==========================+===============================+========================================+
| ``config/S0_10.yaml``    | ``0`` (placebo arm)           | ``[]``                                 |
+--------------------------+-------------------------------+----------------------------------------+
| ``config/S3_10.yaml``    | ``3`` (no BMI gain)           | ``[]``                                 |
+--------------------------+-------------------------------+----------------------------------------+
| ``config/S0_SA_10.yaml`` | ``0`` (placebo arm)           | the six sensitivity analysis variables |
+--------------------------+-------------------------------+----------------------------------------+
| ``config/S3_SA_10.yaml`` | ``3`` (no BMI gain)           | the six sensitivity analysis variables |
+--------------------------+-------------------------------+----------------------------------------+

``S0`` is the baseline arm and ``S3`` the intervention arm; the ``_SA`` variants are the same two
arms with the sensitivity analysis variables perturbed. Every one of the four must set
``replications`` to the same value as ``num_replications`` in the ``Snakefile``, because the
``simulate`` and ``combine`` rules refer to a specific zero-padded replication file
(e.g. ``replication_10/new_init_age.parquet`` for 10 replications) to decide whether the
simulation is done.

To create the configs for a new replication count, copy the existing set and edit them. For a
200-replication run::

    cd config
    for scenario in S0 S3 S0_SA S3_SA; do
        cp ${scenario}_10.yaml ${scenario}_200.yaml
    done

Then in each new file set::

    replications: 200      # must match num_replications in the Snakefile
    num_cpus: 150          # number of dask workers; tune to the machine you are running on

and leave ``bmi_intervention_scenario`` and ``sa_variables`` as they were in the file you copied
from, since those are what distinguish the four scenarios. Finally, update the ``Snakefile`` to
match::

    num_replications = 200

The repository already ships configs for 10, 200, 1000, 2500 and 10000 replications, so in most
cases changing ``num_replications`` to one of those values is all that is needed.

^^^^^^^^^^^^^^^^^^^^^^^^^
Step 2: Run the Snakefile
^^^^^^^^^^^^^^^^^^^^^^^^^
From the project root, do a dry run first to check which jobs Snakemake plans to execute::

    snakemake --dry-run

Then run the full workflow, giving Snakemake the number of cores it may use::

    snakemake --cores 8

Useful variations::

    snakemake --cores 8 --printshellcmds     # echo the shell command for each job
    snakemake --cores 8 --rerun-incomplete   # redo jobs left half-finished by an interrupted run
    snakemake --cores 8 --forcerun simulate  # force the simulations to run again

Note that ``--cores`` bounds how many *jobs* Snakemake runs at once, while ``num_cpus`` in the
config bounds the dask workers *within* a single simulation. The simulations are the expensive
part of the workflow, so on a large shared machine it is normal to keep ``--cores`` small and
``num_cpus`` large.

You can also build a single target instead of the whole workflow by naming a rule or an output
file. For example, to run only the baseline simulation and combine its output::

    snakemake --cores 8 out/S0_10/combined/bmi_int_cascade.parquet

Results land in ``out/S0_{num_replications}/``: ``final_table.csv``, the paper figures
(``fig2a.png`` … ``fig3d.png``) with their accompanying ``figure*_table.csv`` files, and the
sensitivity analysis plots ``tornado_absolute.png`` and ``tornado_relative.png``.

Because the ``simulate`` rule passes ``--overwrite`` to ``2_simulate.py``, re-running a simulation
through Snakemake deletes the existing ``out/{config}`` directory for that scenario. Move or copy
any output you want to keep before re-running.

=========================================
Adding a New Plot Output to the Workflow
=========================================

Snakemake only knows about the files a rule *declares*. A figure that a script happens to write but
that no rule lists as an ``output:`` is invisible to the workflow: it will still appear in ``out/``
when the script runs, but Snakemake will not rebuild it when its inputs change, and asking for it
as a target will fail with ``MissingRuleException``. (``6_bmi_plots.py`` already writes several
such files, e.g. ``figS2.png`` and ``figure2a_table.csv``, which are not declared anywhere in the
``Snakefile``.)

There are therefore two edits to make for any new plot, and a third if you added a new script:

1. **Declare it as an output** of the rule that produces it.
2. **Add it to** ``rule all`` so that a plain ``snakemake`` run actually builds it. ``rule all`` is
   the default target; anything not reachable from it is only built if you name it explicitly on
   the command line.
3. If the plot comes from a *new* script, add a new rule that runs it.

The declared path and the path the script writes to must match exactly, or Snakemake will fail the
job with ``MissingOutputException`` after the script has already succeeded.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Case 1: Adding a plot to an existing plotting script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Suppose you add a new figure to ``scripts/6_bmi_plots.py``. That script receives its output
location as ``--out_dir`` and writes every figure relative to it, so the new plot should be saved
the same way as the existing ones::

    my_fig.savefig(out_dir / "fig4a.png", bbox_inches="tight")
    df.to_csv(out_dir / "figure4a_table.csv")

``6_bmi_plots.py`` is run by the ``bmi_paper_outputs`` rule, which passes
``out_dir = f"out/S0_{num_replications}"``. Add the new files to that rule's ``output:`` list, using
the same ``f"out/S0_{num_replications}/..."`` form as its neighbours so the path tracks the
replication count::

    rule bmi_paper_outputs:
        input:
            ...
        output:
            ...
            f"out/S0_{num_replications}/fig3d.png",
            f"out/S0_{num_replications}/figure3d_table.csv",
            f"out/S0_{num_replications}/fig4a.png",             # new
            f"out/S0_{num_replications}/figure4a_table.csv",    # new
        params:
            ...

Then add the same two paths to the ``input:`` list of ``rule all``. No change to the ``shell:``
command is needed, because the script already writes everything it produces into ``--out_dir``.

If the new plot needs data the rule does not currently read — say a combined parquet that
``bmi_paper_outputs`` does not list — add that file to the rule's ``input:`` as well, so Snakemake
knows to build it first and to redo the plots when it changes.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Case 2: Adding a new plotting script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Write the script so that it takes its inputs and its output directory as command line arguments,
the way ``6_bmi_plots.py`` and ``7_bmi_sa.py`` do::

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline")
    parser.add_argument("--variable")
    parser.add_argument("--out_dir")
    args = parser.parse_args()

This keeps the paths in the ``Snakefile`` rather than hard-coded in the script, which is what lets
the workflow re-point everything at a different replication count by changing one number.

Then add a rule for it. Its ``input:`` should be the combined parquet directories it reads (the
outputs of ``combine`` / ``aggregate``), so that Snakemake runs the simulations and combines their
output before the plots::

    rule my_new_plots:
        input:
            f"out/S0_{num_replications}/combined/bmi_int_cascade.parquet",
            f"out/S3_{num_replications}/combined/bmi_int_cascade.parquet",
        output:
            f"out/S0_{num_replications}/fig4a.png",
            f"out/S0_{num_replications}/figure4a_table.csv",
        params:
            out_dir = f"out/S0_{num_replications}",
            baseline = f"out/S0_{num_replications}/combined",
            variable = f"out/S3_{num_replications}/combined",
        shell:
            "python scripts/9_my_new_plots.py --baseline {params.baseline} "
            "--variable {params.variable} --out_dir {params.out_dir}"

Note the split between ``input:`` and ``params:``: files listed under ``input:`` are what Snakemake
builds and timestamp-checks, while ``params:`` holds the directory strings handed to the script.
The existing rules pass directories through ``params:`` and list the individual parquet files they
depend on under ``input:``; follow that pattern.

Finally add the new outputs to ``rule all``::

    rule all:
        input:
            ...
            f"out/S0_{num_replications}/tornado_relative.png",
            f"out/S0_{num_replications}/fig4a.png",             # new
            f"out/S0_{num_replications}/figure4a_table.csv",    # new

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Checking the new rule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A dry run shows whether Snakemake picked up the new target and what it thinks it needs to build to
get there, without running anything::

    snakemake --dry-run
    snakemake --dry-run out/S0_10/fig4a.png    # just the new plot

Once the simulations have already been run, you can iterate on the plot alone without redoing them.
Snakemake will not re-run a rule whose outputs are newer than its inputs, so force it::

    snakemake --cores 8 --forcerun my_new_plots

If instead you see ``MissingRuleException``, the target you asked for is not declared as the output
of any rule — check for a typo between the ``savefig`` path in the script and the ``output:`` entry
in the ``Snakefile``.
