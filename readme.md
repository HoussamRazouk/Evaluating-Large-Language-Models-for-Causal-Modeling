
# Introduction  
This code investigates the effectiveness of large language models (LLMs) in transforming causal domain knowledge into a representation that better aligns with recommendations from causal data science.

## method
The approach consists of two main steps. 
### Nodes in the causal diagram represents causal variables not its realized values
The first step involves identifying if two entries, gathered through brainstorming approaches such as using cognitive maps or extracted using automated causal information extraction method, describe the values of the same causal variable.
Our method investigates whether giving two texts, which could represent a cause or effect of different causal relations, actually represents different values of the same causal variable. To test this, a prompt has been designed for the purpose. 
To compare and test different LLMs' effectiveness in achieving these tasks, another prompt has been utilized to generate data. The generated data consists of a specified number of causal variables for a selected domain and several examples of their realized values. 
The LLMs have been instructed to provide examples of realized values using noun phrases and to avoid ambiguous values such as 'high' or 'low,' as well as numerical values. 
The generated data are then sampled for positive and negative examples based on the generated models and the domain. 
A pair of two texts is considered to represent different values of the same causal variable if they occurred under the variable example of the generated lists. 
