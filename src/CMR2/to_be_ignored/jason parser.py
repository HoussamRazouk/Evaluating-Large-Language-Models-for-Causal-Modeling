import json
import pandas as pd 
import sys
sys.path.append('.')
from src.CMR2.config import conf_init
a={
    "Interaction Events": 
    [
        {
            "Interaction value": "Presence of public art installations",
            "Variable values": 
            [
                {
                    "Variable definition": "Public space quality",
                    "Variable value": "Enhanced",
                },
                {
                    "Variable definition": "Community engagement",
                    "Variable value": "Increased",
                }
            ],
            "Explanation": "Public art installations can improve the quality of public spaces, making them more attractive and welcoming. Additionally, they can stimulate community engagement by providing a platform for expression and interaction."
        },
        {
            "Interaction value": "High density of coffee shops",
            "Variable values": 
            [
                {
                    "Variable definition": "Economic vitality",
                    "Variable value": "Boosted",
                },
                {
                    "Variable definition": "Walkability",
                    "Variable value": "Improved",
                }
            ],
            "Explanation": "A high density of coffee shops can indicate a strong local economy, as these businesses often thrive in areas with high foot traffic. Additionally, the presence of coffee shops can encourage walking, as people may visit these establishments for meetings, study sessions, or casual gatherings."
        },
        {
            "Interaction value": "Implementation of bike-sharing programs",
            "Variable values": 
            [
                {
                    "Variable definition": "Sustainable transportation",
                    "Variable value": "Promoted",
                },
                {
                    "Variable definition": "Urban mobility",
                    "Variable value": "Enhanced",
                }
            ],
            "Explanation": "Bike-sharing programs can encourage sustainable transportation by providing an eco-friendly alternative to cars. Furthermore, they can improve urban mobility by offering a flexible and convenient way to navigate the city."
        },
        {
            "Interaction value": "Proliferation of coworking spaces",
            "Variable values": 
            [
                {
                    "Variable definition": "Economic diversity",
                    "Variable value": "Expanded",
                },
                {
                    "Variable definition": "Innovation ecosystem",
                    "Variable value": "Strengthened",
                }
            ],
            "Explanation": "The growth of coworking spaces can contribute to economic diversity by accommodating various industries and professionals. Additionally, it can foster a stronger innovation ecosystem by bringing together entrepreneurs, freelancers, and startups under one roof."
        },
        {
            "Interaction value": "Availability of green roofs and walls",
            "Variable values": 
            [
                {
                    "Variable definition": "Urban heat island effect",
                    "Variable value": "Mitigated",
                },
                {
                    "Variable definition": "Biodiversity",
                    "Variable value": "Enhanced",
                }
            ],
            "Explanation": "Green roofs and walls can help combat the urban heat island effect by reducing temperatures in built-up areas. Furthermore, they can contribute to increased biodiversity by providing habitats for various plant and animal species."
        },
        {
            "Interaction value": "High concentration of cultural institutions",
            "Variable values": 
            [
                {
                    "Variable definition": "Cultural vitality",
                    "Variable value": "Elevated",
                },
                {
                    "Variable definition": "Tourist appeal",
                    "Variable value": "Increased",
                }
            ],
            "Explanation": "A high concentration of cultural institutions, such as museums, theaters, and galleries, can signify a vibrant cultural scene. Additionally, it can attract tourists by offering a rich array of artistic and historical experiences."
        },
        {
            "Interaction value": "Widespread use of smart city technologies",
            "Variable values": 
            [
                {
                    "Variable definition": "Efficient resource management",
                    "Variable value": "Optimized",
                },
                {
                    "Variable definition": "Resident satisfaction",
                    "Variable value": "Improved",
                }
            ],
            "Explanation": "The widespread use of smart city technologies, such as sensors and data analytics, can lead to more efficient resource management by optimizing services like waste collection, traffic management, and energy consumption. Moreover, it can enhance resident satisfaction by providing convenient and responsive urban services."
        },
        {
            "Interaction value": "Prevalence of adaptive reuse projects",
            "Variable values": 
            [
                {
                    "Variable definition": "Historic preservation",
                    "Variable value": "Respected",
                },
                {
                    "Variable definition": "Sustainable development",
                    "Variable value": "Promoted",
                }
            ],
            "Explanation": "The prevalence of adaptive reuse projects, which involve repurposing existing buildings for new uses, can demonstrate a commitment to historic preservation. Furthermore, it can promote sustainable development by reducing the demand for new construction materials and preserving the embodied energy of existing structures."
        },
        {
            "Interaction value": "Robust public transportation network",
            "Variable values": 
            [
                {
                    "Variable definition": "Air quality",
                    "Variable value": "Improved",
                },
                {
                    "Variable definition": "Social equity",
                    "Variable value": "Advanced",
                }
            ],
            "Explanation": "A robust public transportation network can contribute to improved air quality by reducing the number of cars on the road. Additionally, it can promote social equity by providing affordable and accessible transportation options for all residents, regardless of income or ability."
        },
        {
            "Interaction value": "Integration of green infrastructure",
            "Variable values": 
            [
                {
                    "Variable definition": "Climate change resilience",
                    "Variable value": "Strengthened",
                },
                {
                    "Variable definition": "Liveability",
                    "Variable value": "Enhanced",
                }
            ],
            "Explanation": "The integration of green infrastructure, such as parks, wetlands, and green corridors, can enhance climate change resilience by mitigating the impacts of extreme weather events and flooding. Furthermore, it can improve liveability by providing opportunities for recreation, relaxation, and social interaction."
        }
    ]
}
config=conf_init()

CMR2_generated_data_dir=config['CMR2_generated_data_dir']

model='mixtral-8x7b-instruct'
domain='urban studies'

df=pd.DataFrame(a["Interaction Events"])
df.to_csv(CMR2_generated_data_dir+f'CMR2_Generated_data_{model}_{domain}.csv',index=False) 
df.to_pickle(CMR2_generated_data_dir+f'CMR2_Generated_data_{model}_{domain}.pkl') 
print (f'succeed: : {model} {domain}')

#print(a[250:350])

#response=json.loads(a[a.index('\n'):])

#d={"Variable value": "Enhanced",}
#d["Variable value"]