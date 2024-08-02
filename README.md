## Collaborative Adaptive Learning

Our work presents solutions for test assignments for groups of learners in educational systems which are based on three dimensions:
- Group Learning: It captures peer learning in groups of learners.
- Expected Performance: It is the expected performance of a group of learners for an assigned test.
- Aptitude: It ensures that each learner in a group is challenged by at least one test.

## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have python3 installed
* You have R installed
* You have installed requirements  `pip install -r requirements.txt`
* You have jupyter notebook installed

## Reproducibility of Experiments
To reproduce the reported results:
* Make sure that data files are imported in ```./Data```.
* Generate the groups of learners and the batches that are assigned to each group by running the file `groupingAssis2009.py` for the ASSISTment 2009 dataset (`groupingAssisChallenge.py` for the ASSISTment Challenge dataset). This creates files within ```./res_Assis2009``` (```./res_Assis``` for ASSISTment Challenge). The directory ```./res_Assis2009/groupings``` contains the file of groups and assigned tests. The directories ```./res_Assis2009/learners``` and ```./res_Assis2009/tests``` contain the initial knowledge of learners and the difficulty of tests for all skills respectively. ```./res_Assis2009/perfs``` contains the files of the expected performances of learners.
* After generating the groups and the batches, run `getResultsAssis2009.py` (or `getResultsAssisChallenge.py` for ASSISTment Challenge) to generate the metrics reported in the paper. This generates a csv file in ```./res_Assis2009/results```.
* Use the notebook `graphs.ipynb` to generate the figures.
