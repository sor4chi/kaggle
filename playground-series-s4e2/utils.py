import constant as const
import pyjson5

# Gender: ['Male', 'Female']
# family_history_with_overweight: ['yes', 'no']
# FAVC: ['yes', 'no']
# CAEC: ['Sometimes', 'Frequently', 'no', 'Always']
# SMOKE: ['no', 'yes']
# SCC: ['no', 'yes']
# CALC: ['Sometimes', 'no', 'Frequently']
# MTRANS: ['Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike']
# NObeyesdad: ['Overweight_Level_II', 'Normal_Weight', 'Insufficient_Weight', 'Obesity_Type_III', 'Obesity_Type_II', 'Overweight_Level_I', 'Obesity_Type_I']

def get_mapping() -> dict:
    MAPPING = {}
    with open(const.MAPPING_CONFIG_PATH, 'r') as f:
        MAPPING = pyjson5.load(f)
    return MAPPING
