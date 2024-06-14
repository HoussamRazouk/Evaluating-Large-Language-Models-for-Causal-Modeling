
def conf_init():
    """
        Summary: This function initializes all the hyperparameters used in the experiment. 
        These parameters include:
        1. Domains: The domains where the data will be generated and tested.
        2. Models: The different models that are being tested in the experiment.
        3. Number of variables: The quantity of variables to be generated for the experiment. 
        4. Number of positive samples: The quantity of positive samples to be sampled for the experiment.
        5. Number of negative samples: The quantity of negative samples to be sampled for the experiment.
    """ 

    
    ### the domains which the data set will be generated for 
    domains=["urban studies",
             "physics",
             "health",
             "semiconductor manufacturing",
             "computer science",
             "sociology",
             "psychology",
             "finance"]

    models=["llama3-70b",
            "mixtral-8x22b-instruct",
            "gpt-3.5-turbo",
            "gpt-4-turbo",
            "llama3-8b",
            "mixtral-8x7b-instruct",
            "mistral-7b-instruct"
            ]
    
    number_variables=10
    number_of_positive_samples=10
    number_of_negative_samples=10
    CMR1_generated_data_dir="results/CMR1/CMR1_generated_data/"
   
    CMR1_sample_data_file="results/CMR1/sampled_data/sampled_data_set_large.csv"
    CMR1_evaluated_data_dir="results/CMR1/evaluated_data/"
    
    
    
    return {
        'domains':domains,
        'models':models,
        'number_variables':number_variables,
        'number_of_positive_samples':number_of_positive_samples,
        'number_of_negative_samples':number_of_negative_samples,
        'CMR1_generated_data_dir':CMR1_generated_data_dir,
        'CMR1_sample_data_file':CMR1_sample_data_file,
        'CMR1_evaluated_data_dir':CMR1_evaluated_data_dir,  
    }