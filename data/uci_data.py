import pandas as pd

DATASET_PATH = "data/heart_disease/heart_disease_uci.csv"

class UCIData():
    def __init__(self):
        self.dataset = pd.read_csv(DATASET_PATH)
        print(f"Loaded UCI Heart Disease dataset with {self.dataset.shape[0]} rows and {self.dataset.shape[1]} columns.")
        
    def get_data(self):
        return self.dataset
    
    def get_features(self) -> pd.DataFrame:
        return self.dataset.drop(columns=['num'])

    def get_targets(self) -> pd.DataFrame:
        return self.dataset.data['num']
    
    def get_headers(self) -> list:
        return self.dataset.columns.to_list()
    
    


'''
## Fetch returns
- **`dataset`**
	- **`data`**: Contains dataset matrices as **pandas** dataframes
		- `ids`: Dataframe of ID columns
		- `features`: Dataframe of feature columns
		- `targets`: Dataframe of target columns
		- `original`: Dataframe consisting of all IDs, features, and targets
		- `headers`: List of all variable names/headers
	
    - **`metadata`**: Contains metadata information about the dataset
		- See Metadata section below for details
	
    - **`variables`**: Contains variable details presented in a tabular/dataframe format
		- `name`: Variable name
		- `role`: Whether the variable is an ID, feature, or target
		- `type`: Data type e.g. categorical, integer, continuous
		- `demographic`: Indicates whether the variable represents demographic data
		- `description`: Short description of variable
		- `units`: variable units for non-categorical data
		- `missing_values`: Whether there are missing values in the variable's column


## Metadata
- `uci_id`: Unique dataset identifier for UCI repository 
- `name`
- `abstract`: Short description of dataset
- `area`: Subject area e.g. life science, business
- `task`: Associated machine learning tasks e.g. classification, regression
- `characteristics`: Dataset types e.g. multivariate, sequential
- `num_instances`: Number of rows or samples
- `num_features`: Number of feature columns
- `feature_types`: Data types of features
- `target_col`: Name of target column(s)
- `index_col`: Name of index column(s)
- `has_missing_values`: Whether the dataset contains missing values
- `missing_values_symbol`: Indicates what symbol represents the missing entries (if the dataset has missing values)
- `year_of_dataset_creation`
- `dataset_doi`: DOI registered for dataset that links to UCI repo dataset page
- `creators`: List of dataset creator names
- `intro_paper`: Information about dataset's published introductory paper
- `repository_url`: Link to dataset webpage on the UCI repository
- `data_url`: Link to raw data file
- `additional_info`: Descriptive free text about dataset
	- `summary`: General summary 
	- `purpose`: For what purpose was the dataset created?
	- `funding`: Who funded the creation of the dataset?
	- `instances_represent`: What do the instances in this dataset represent?
	- `recommended_data_splits`: Are there recommended data splits?
	- `sensitive_data`: Does the dataset contain data that might be considered sensitive in any way?
	- `preprocessing_description`: Was there any data preprocessing performed?
	- `variable_info`: Additional free text description for variables
	- `citation`: Citation Requests/Acknowledgements
 - `external_url`: URL to external dataset page. This field will only exist for linked datasets i.e. not hosted by UCI
  
'''