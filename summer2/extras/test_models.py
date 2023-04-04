from summer2 import CompartmentalModel
from summer2.parameters import Parameter as param
from summer2.parameters import DerivedOutput

def sir():
    m = CompartmentalModel([0,100], ["S","I","R"], "I")
    m.set_initial_population({"S": 990.0, "I": 10.0})
    m.add_infection_frequency_flow("infection", param("contact_rate"), "S", "I")
    m.add_transition_flow("recovery", param("recovery_rate"), "I", "R")
    m.request_output_for_flow("incidence", "infection")
    m.set_default_parameters({"contact_rate": 0.4, "recovery_rate": 0.1})
    return m

