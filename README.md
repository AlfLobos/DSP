# DSP

This github contain the code from the paper: "Optimal Bidding, Allocation, and Budget Spending for a Demand Side Platform with Generic Auctions", Paul Grigas, Alfonso Lobos, Zheng Wen, Kuang-chih Lee . 

This github contains two main folders having the code for the arificial experiments and those based on the Criteo Attribution dataset https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/.

Our code requires:
1) Python 3.5 or higher.
2) Access to a Gurobi license.
3) Jupyter-notebook working with python 3.5 for the artificial experiment.
4) Access to a unix/linux server for the real-data experiment. Between tunning the constant for the dual step size and then running the experiments the amount of computation exceeded what was obtainable by a personal computer on a reasonable amount of time. For that reason we use the engineering cluster Savio at UC, Berkeley to run the experiments (https://research-it.berkeley.edu/services/high-performance-computing/user-guide/savio-user-guide). More details on how this code was run later.

## Folder: ArtificialExperiment

Find here the code for the artificial experiment. This code was simple enough that it run using jupyter notebook in a standard laptop (in our case a 2015 mac book pro). In terms of files and folders we have:

### Python Files and Internal Folders
- DataCreation.py: It has functions needed to create the data used in this experiment. 
- RhoAndBeta.py: File that precises how to create the bid landscape parameters for the synthetic experiment. 
- UtilitiesOptimization.py: This file has the following functions:
    - Functions to call the phase primal dual scheme shown in the paper. There are two methods (excluding the primal recovery step). 'SubgrAlgSavPrimDualObjFn_L2Ind' is used to run the two phase scheme using the budget constraint utility function, and 'SubgrAlgSavPrimDualObjFn_L2Ind' when either the utility function function budget contraints + target spending or target spending is used.
    - Functions to perform the primal recovery step by using the Gurobi solver. In particular, 'CalculateLPGurobi' performs the primal recovery step for the budget constraints utility function, and 'CalculateQuadGurobi' for the budget constraints plus target spending and target spending functions. 
    - Functions to precise the form of the gradient for the different utility functions. 
    - Form of the optimal bids vector for both first and second price auctions.
- SimulationCode.py: This file has the code of the simulator used to simulate a real operation. It has methods to run the pareto and budget sensitivity experiments.
- (Folder) JupyterNotebooks: As the name implies it contains jupyter notebooks to run to experiments of this section. We have divided the files depending if they run using first or second-price mechanisms and if it is the Pareto or sensitivity to budget experiment.
- (Folder) ResultsPareto: Contains .csv files of the results of running the Pareto experiment. 'FP' and 'SP' correspond to first and second-price auctions and 'Ind' is the results when we run the experiment using the target budget constraint utility function. We have run this utility function separately as we do only need to run once, while the others and the greedy heuristic are run using different combination of parameters.
- (Folder) ResultsSensitivity: Contains .csv files of the results of running the sensitivity to budget experiment. 'FP' and 'SP' correspond to first and second-price auctions.
- (Folder) Instance Info: Contains the files produced when running DataCreation.py. 


## Folder: Criteo

Find here the code for the experiment using the Criteo Attribution dataset https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/. This code is considerably harder to run than the previous, and we used the Savio cluster at UC, Berkeley to run it (read the introduction). This folder contains two main subfolders, the first dedicated to creating the data for the experiment, and the second to run the experiment. 

### Folder: CreateData

We thank Andrew Ding for helping with this part of the experiment. Criteo's data is composed of 17.5 million rows of bidding logs performed by the company on behalf of 700 campaigns (the data uses 2.4 Gb when uncompressed). Each bidding log indicates the campaign for which Criteo bid on behalf, nine categorical impression features, if a click occurred, the price paid by Criteo to the Ad-exchange (a transformed version of it), and other columns we do not use. There are three main tasks on how to create the data for this experiment. 1. Developing an algorithm to associate each bidding log to an impression type, in specific, how to use the nine categorical features to determine a number representing the impression type. 2. Creating the bid landscape functions. 3. Creating every other aspect of the data, *e.g.*, creating the campaigns budgets, expected revenue terms, expected number of impressions to be received of each type, etc. Is important to mention, that we split the bidding logs in the first 75\% used to create the training and validation data, and the last 25\% of the logs to create the test data. The bidding logs are sorted by date, and they correspond to four weeks of bidding data performed by Criteo (the data includes a time_stamp feature that is a number that relates to the time when Criteo bid, but not a proper date column). We have created a google drive https://drive.google.com/open?id=1pchD3N91uFPzNOzD7lsCHkSUXhOFnZbA with intermediate results obtained after running our algorithm for impression types. We have done this as one of these files weights 2.99Gb and other 0.8 Gb. We have also included in the drive a folder called 'DataCriteo' that contains the final files used to run the experiment (in case you want to skip the data creation part). Is important to say that the data creation for this experiment was a computationally expensive process, in particular, the impression type creation.  We now proceed to comment on how the data creation was done.


**Creating Impression Types and Some Other Data**


The impression types were obtained running CART. The click column was used as the label column, the nine categorical columns as features, and Gini as the impurity function. To find an adequate complexity to use, we first tried 100 possible complexity parameters. Each run creates different trees which can be used to associate a bidding log to an impression type. Of those 100, we selected eight candidates that had an appropriate trade-off between Validation Accuracy and Gini impurity values. Of these eight we end up choosing one which had a number of impression types that was computationally tractable for our experiments.  This code was done in R and the files used were 'Criteo-Cart-1.R', 'Bulk.R', and 'Bulk2.R' respectively.


**Rest of the Data (Including Impression Types)**

Please download the files in https://drive.google.com/open?id=1pchD3N91uFPzNOzD7lsCHkSUXhOFnZbA before proceeding to this part. The data in the drive folder are:
- campaignBidRecords.csv: For each campaign, we counted the total amount of bid logs in which it appears and how many of those logs were in the first three weeks and last week of the data (first 75\% and last 25\% of the data resp.). (In our experiment we do not use campaigns with less than 200 bid logs in either the train or test set).
- train_with_clusters: The first 75\% of the rows from the Criteo data but in which we have added eight columns. These columns represent the impression types from the candidate eight complexity parameters used in CART. Of those eight columns, we used the impression types shown in the 'cluster76' to run our algorithm.
- test_with_clusters: Same as the above but for the test data.
- train76.csv: The rows represent impression types and columns represent a given campaign, except for the second, which shows the number of the leaf representing assigned by the CART algorithm to the impression type. (An impression type has 1:1 correspondence to a leaf in the CART tree.) The triplets that appear in the document represent the number of observed bidding logs, clicks, and conversions for a given pair of impression type and campaign. As the name of the file implies, this table is built only on training data.
- test76.csv: Analogous to the previous except that test data is used.
- train76ModWithoutCampaignLabels.csv and test76ModWithoutCampaignLabels.csv:  A re-formatting of the train76.csv and test76.csv files, as the triplet format was tricky to read using pandas.
- (Folder) DataCriteo: The final results when we run a Jupiter-notebook commented below.

To obtain all the necessary data, copy the files commented above in a folder called 'Criteo' and run the Jupyter-notebook 'CreatingDataForCriteo.ipynb' (which needs to be in the same folder as Criteo). All the data to be used by our algorithm is saved using the Pickle format (we create more data than the one used in the experiment). As an important remark, we create the bidding landscape functions using the empirical distribution for the impression types. In comparison, our first approach was to fit a logNormal or Beta distribution to all the market price values of a given impression type to obtain the $\rho_i(\cdot)$ functions. We obtained poor results by doing this as the data seems to has a small reserve price which accounted for some impression types to 10\% or more of the observed market price values. Then, curve fitting performs poorly for unimodular continuous distributions. Given that there are at least a couple of thousands of market price values per impression type using the empirical distribution of the data is fine. For estimating $\beta_i(b)$ we can take the average of all market prices less than $b$, and for $\rho_i(b)$ the proportion of market prices that are below $b$ with respect to the total. To make the estimation faster, we interpolated the data. For example, we saved the bid values that leave $(i/300)\%$ of the market prices below for $i \in \{1,\dots,300\}$. Then, $\rho_i(b)$ can be calculated as a simple bisection method and interpolating at the end between the closest two points (most likely $b$ will not match any number exactly). For $\beta(\cdot)$ the trick was similar. Here, we would like to comment of a few pickle files of interest:
- ImpInOrder: Impression types in order as they appear in the test data (training does not need this information).
- MPInOrder: Market Price in order as they appear in the test data (training does not need this information).
- vector_m: The budgets of the campaigns
- PPFLists: Lists with the 300 data points to calculate $\rho_i(\cdot)$ for the different impression types.
- numericBeta: Matrix used to calculate $\beta_i(\cdot)$.
- vector_r: This vector is a little confusing as it corresponds to the price paid by the campaigns when a click of interest occurs (while in the paper 'r' is used to represent the expected revenue which typically in the previous amount times the click-through rate). Is important to say that this quantity is later modified when we run the experiments by multiplying it by a fixed perturbation vector for all experiments. 

### Folder: RunExperiment


In the folder, you can find three files 'Utilitites.py', 'UtilitiesOptimization.py', and 'SimulationCode.py' that roughly have the same functions as the ones already explained for the Artificial Experiment. Also, there are three subfolders called 'PyFilesFromServer', 'Tasks', 'Results'. Before going into deeper details, we run three experiments which differ on the expected revenue vector used. In particular, we multiply the revenue vector by  0.5, by 0.75, and leave it unchanged, which correspond to the inner strings '05', '075', and 'FB' respectively in the files' names. As a general comment, we use the Savio cluster by running each penalization parameter configuration in a different core (we use nodes with up to 24 cores). Each work saves an independent '.csv' file, for that reason we have added the 'Results' folder in which we aggregate the results obtained in those hundreds of .csv files (and we perform some analysis on them too).

** (Folder) PyFilesFromServer **

All files having 'Extra' on its name mean that we tried more penalization parameters than the ones we thought we were going to need. Also, we have decided to run the utility function 'budget constraints and target spending' using separate files as that utility function only needs to be run once. The configuration to be run is given when we run the file using the Python 'argparse' library. All different jobs to be run are in the files in the 'Tasks' folder. The configurations we need to run are two. 1. The hyperparameter search for the subgradient algorithm. 2. The penalization parameter for the Pareto method.


** (Folder) Tasks **


It contains files with lines of the form
..
../CriteoParetoFBEmp.py --id 7
../CriteoParetoFBEmp.py --id 8
../CriteoParetoFBEmp.py --id 9
..
where '--id 7' means to run the parameter configuration number 7 on a given array. Each line is a different job that can be run in parallel on different cores. Please change '..' for wherever the Python files from the PyFilesFromServer folder are located.


** (Folder) Results **

Contains six files in which '05', '075', and 'FB' show the revenue vector used (explained at the beginning of this subsection), and 'Val' represents the results from the "tunning" experiments. These files contain the aggregated information from the many files generated when running each of jobs explained of the files in Tasks folder. 
