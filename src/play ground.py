

import pandas as pd
#CMR1_Generated_data_llama3-70b_health
a=[{'variable definition': 'Body Mass Index', 'values': ['Obesity', 'Overweight', 'Normal Weight', 'Underweight']}, {'variable definition': 'Blood Pressure', 'values': ['Hypertension', 'Prehypertension', 'Normal Blood Pressure', 'Hypotension']}, {'variable definition': 'Sleep Quality', 'values': ['Insomnia', 'Sleep Apnea', 'Restless Leg Syndrome', 'Good Sleep']}, {'variable definition': 'Dietary Habits', 'values': ['Vegetarian', 'Vegan', 'Omnivore', 'Junk Food Addict']}, {'variable definition': 'Physical Activity Level', 'values': ['Marathon Runner', 'Gym Enthusiast', 'Casual Walker', 'Couch Potato']}, {'variable definition': 'Mental Health Status', 'values': ['Depression', 'Anxiety Disorder', 'Bipolar Disorder', 'Mental Wellness']}, {'variable definition': 'Smoking Status', 'values': ['Heavy Smoker', 'Social Smoker', 'Ex-Smoker', 'Non-Smoker']}, {'variable definition': 'Alcohol Consumption', 'values': ['Heavy Drinker', 'Social Drinker', 'Light Drinker', 'Teetotaler']}, {'variable definition': 'Family Medical History', 'values': ['Heart Disease', 'Diabetes', 'Cancer', 'No Known Conditions']}, {'variable definition': 'Occupational Hazards', 'values': ['Toxic Chemical Exposure', 'Heavy Lifting', 'Sedentary Job', 'Safe Work Environment']}]
df=pd.DataFrame()
df=df.append(a, ignore_index=True)
df.to_csv(f'results/CMR1_Generated_data_llama3-70b_health.csv',index=False)

#CMR1_Generated_data_llama3-70b_physics
a=[{'variable definition': 'Type of Radiation', 'values': ['Gamma Rays', 'X-Rays', 'Alpha Particles', 'Beta Particles', 'Neutron Beams']}, {'variable definition': 'State of Matter', 'values': ['Solid Ice', 'Liquid Water', 'Gaseous Vapor', 'Plasma State', 'Bose-Einstein Condensate']}, {'variable definition': 'Celestial Body', 'values': ['Dwarf Planet', 'Gas Giant', 'Terrestrial Planet', 'Neutron Star', 'Black Hole']}, {'variable definition': 'Force of Nature', 'values': ['Electromagnetic Force', 'Strong Nuclear Force', 'Weak Nuclear Force', 'Gravitational Force']}, {'variable definition': 'Particle Accelerator', 'values': ['Linear Accelerator', 'Cyclotron', 'Synchrotron', 'Collider', 'Storage Ring']}, {'variable definition': 'Thermodynamic Process', 'values': ['Isothermal Expansion', 'Adiabatic Compression', 'Isobaric Heating', 'Isochoric Cooling', 'Cyclic Process']}, {'variable definition': 'Optical Phenomenon', 'values': ['Total Internal Reflection', 'Diffraction Pattern', 'Interference Fringes', 'Polarization', 'Scattering']}, {'variable definition': 'Quantum Mechanical System', 'values': ['Harmonic Oscillator', 'Infinite Square Well', 'Finite Potential Barrier', 'Hydrogen Atom', 'Quantum Harmonic Oscillator']}, {'variable definition': 'Astronomical Event', 'values': ['Solar Eclipse', 'Lunar Eclipse', 'Supernova Explosion', 'Gravitational Wave Detection', 'Comet Impact']}, {'variable definition': 'Electrical Circuit Element', 'values': ['Resistor', 'Inductor', 'Capacitor', 'Diode', 'Transformer']}, {'variable definition': 'Nuclear Reaction', 'values': ['Fission Reaction', 'Fusion Reaction', 'Radioactive Decay', 'Neutron-Induced Reaction', 'Proton-Induced Reaction']}]
df=pd.DataFrame()
df=df.append(a, ignore_index=True)
df.to_csv(f'results/CMR1_Generated_data_llama3-70b_physics.csv',index=False)

#CMR1_Generated_data_llama3-70b_semiconductor manufacturing

a=[{'variable definition': 'Wafer Type', 'values': ['Silicon Wafer', 'Germanium Wafer', 'Gallium Arsenide Wafer', 'Indium Phosphide Wafer', 'Silicon Carbide Wafer']}, {'variable definition': 'Fabrication Node', 'values': ['7nm FinFET', '10nm FinFET', '14nm FinFET', '28nm HKMG', '40nm LP']}, {'variable definition': 'Transistor Type', 'values': ['NMOS Transistor', 'PMOS Transistor', 'Bipolar Junction Transistor', 'Field-Effect Transistor', 'Insulated Gate Bipolar Transistor']}, {'variable definition': 'Dielectric Material', 'values': ['Silicon Dioxide', 'Silicon Nitride', 'Aluminum Oxide', 'Hafnium Oxide', 'Zirconium Oxide']}, {'variable definition': 'Metallization Layer', 'values': ['Aluminum Interconnect', 'Copper Interconnect', 'Tungsten Interconnect', 'Copper-Aluminum Hybrid Interconnect', 'Copper-Tungsten Hybrid Interconnect']}, {'variable definition': 'Etching Method', 'values': ['Wet Etching', 'Dry Etching', 'Reactive Ion Etching', 'Deep Reactive Ion Etching', 'Inductively Coupled Plasma Etching']}, {'variable definition': 'Doping Material', 'values': ['Boron Dopant', 'Phosphorus Dopant', 'Arsenic Dopant', 'Antimony Dopant', 'Bismuth Dopant']}, {'variable definition': 'Wafer Cleaning Method', 'values': ['RCA Cleaning', 'Piranha Cleaning', 'SC1 Cleaning', 'SC2 Cleaning', 'HF Last Cleaning']}, {'variable definition': 'Packaging Type', 'values': ['Dual In-Line Package', 'Quad Flat Package', 'Ball Grid Array Package', 'Land Grid Array Package', 'Chip Scale Package']}, {'variable definition': 'Testing Method', 'values': ['Functional Testing', 'Parametric Testing', 'Scan-Based Testing', 'Built-In Self-Test', 'JTAG Testing']}]

df=pd.DataFrame()
df=df.append(a, ignore_index=True)
df.to_csv(f'results/CMR1_Generated_data_llama3-70b_semiconductor manufacturing.csv',index=False)


#CMR1_Generated_data_mixtral-8x22b-instruct_urban studies
a={
  "Variables": [
    {
      "variable definition": "Population density",
      "values": ["sparse suburban", "dense urban", "rural countryside", "compact city center", "industrial zone", "residential area", "commercial district", "historic downtown", "university campus", "waterfront development"]
    },
    {
      "variable definition": "Transportation infrastructure",
      "values": ["extensive subway system", "limited bus network", "bike-friendly city", "car-centric layout", "walkable neighborhoods", "light rail network", "ferry connections", "airport hub", "intercity train station", "gridlocked traffic"]
    },
    {
      "variable definition": "Housing types",
      "values": ["high-rise apartments", "single-family homes", "townhouses", "loft conversions", "historic homes", "modernist architecture", "prefabricated housing", "social housing projects", "luxury condos", "tiny homes"]
    },
    {
      "variable definition": "Green spaces",
      "values": ["abundant parks", "community gardens", "urban forests", "green roofs", "riverfront promenades", "brownfield redevelopment", "greenbelts", "wildlife corridors", "vertical gardens", "urban agriculture"]
    },
    {
      "variable definition": "Cultural amenities",
      "values": ["world-class museums", "vibrant music scene", "historic landmarks", "public art installations", "ethnic enclaves", "creative districts", "festivals and events", "culinary destinations", "theater and dance venues", "literary hubs"]
    },
    {
      "variable definition": "Economic base",
      "values": ["financial services hub", "manufacturing center", "research and development cluster", "tourism destination", "education and healthcare hub", "logistics and distribution hub", "creative industries hub", "government and public sector hub", "energy and resources hub", "agriculture and food processing hub"]
    },
    {
      "variable definition": "Social diversity",
      "values": ["ethnically diverse neighborhoods", "gentrifying areas", "immigrant communities", "student populations", "retirement communities", "family-friendly areas", "artist enclaves", "professional class neighborhoods", "working-class areas", "wealthy enclaves"]
    },
    {
      "variable definition": "Environmental sustainability",
      "values": ["zero-waste initiatives", "renewable energy projects", "green building standards", "carbon-neutral developments", "eco-districts", "sustainable transportation options", "water conservation measures", "urban heat island mitigation", "resilient infrastructure", "climate adaptation strategies"]
    },
    {
      "variable definition": "Governance and planning",
      "values": ["participatory budgeting", "smart city initiatives", "inclusive zoning policies", "transit-oriented development", "historic preservation efforts", "public-private partnerships", "community land trusts", "form-based codes", "new urbanism principles", "mixed-use developments"]
    },
    {
      "variable definition": "Quality of life",
      "values": ["high livability rankings", "affordable cost of living", "safe neighborhoods", "accessible healthcare", "quality education", "recreational opportunities", "cultural vibrancy", "walkability", "public transportation access", "social cohesion"]
    }
  ]
}


df=pd.DataFrame()
df=df.append(a['Variables'], ignore_index=True)
df.to_csv(f'results/CMR1_Generated_data_mixtral-8x22b-instruct_urban studies.csv',index=False)
