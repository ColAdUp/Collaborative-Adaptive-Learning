## Collaborative Adaptive Learning

Our work presents solutions for combining adaptive learning and collaborative learning using test assignments for groups of learners in educational systems which are based on three dimensions:
- Group Learning: It captures peer learning in groups of learners.
- Expected Performance: It is the expected performance of a group of learners for an assigned test.
- Learning Potential: It ensures that each learner in a group is challenged by at least one test.

## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have python3 installed
* You have installed requirements  `pip install -r requirements.txt`
* You have R installed
* You have Anticlust R library installed `install.packages("anticlust")`
* You have Jupyter notebook installed

## Reproducibility of Experiments
To reproduce the reported results:
* Make sure that data files are imported in ```./Data```.
* Generate the groups of learners and the batches that are assigned to each group by running the file `createGroups.py`. Use argument `-d` to specify a dataset from ['Assis2009', 'Neurips', 'AssisChallenge']. This creates files within ```./res_Assis2009``` for the ASSISTment 2009 dataset (```./AssisChallenge``` for ASSISTment Challenge, ```./res_Neurips``` for NeurIPS Challenge). The directory ```./res_Assis2009/groupings``` contains the file of groups and assigned tests. The directories ```./res_Assis2009/learners``` and ```./res_Assis2009/tests``` contain the initial knowledge of learners and the difficulty of tests for all skills respectively. ```./res_Assis2009/perfs``` contains the files of the expected performances of learners.
* After generating the groups and the batches, run `getResults.py` with the same argument `-d` to generate the metrics reported in the paper. This generates a csv file in ```./res_Assis2009/results``` for the ASSISTment 2009 dataset for example.
* Use the notebook `graphs.ipynb` to generate the figures.
