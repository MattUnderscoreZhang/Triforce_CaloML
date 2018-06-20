class energy_filter():
    '''
    filters input event_data based on energy range
    modifies input event_data in place
    '''
    def __init__(self, minEnergy = -1, maxEnergy = 9999):
        self.minEnergy = minEnergy
        self.maxEnergy = maxEnergy
        self.featuresUsed = ['energy']
    def filter(self, event_data):
        passing_indices = (event_data['energy'] > self.minEnergy) & (event_data['energy'] < self.maxEnergy)
        for key,data in event_data.items():
            event_data[key] = data[passing_indices]

class hOverE_filter():
    '''
    filters input event_data based on HCAL/ECAL ratio
    modifies input event_data in place
    '''
    def __init__(self, maxhOverE = 9999):
        self.maxhOverE = maxhOverE
        self.featuresUsed = ['HCAL_ECAL_ERatio']
    def filter(self, event_data):
        passing_indices = (event_data['HCAL_ECAL_ERatio'] < self.maxhOverE) 
        for key,data in event_data.items():
            event_data[key] = data[passing_indices]

class recoOverGen_filter():
    '''
    filters input event_data based on raw reco ECAL+HCAL / gen energy ratio
    modifies input event_data in place
    '''
    def __init__(self, minRecoOverGen = -1):
        self.minRecoOverGen = minRecoOverGen
        self.featuresUsed = ['energy','ECAL_E','HCAL_E']
    def filter(self, event_data):
        passing_indices = (event_data['ECAL_E'] + event_data['HCAL_E'] > self.minRecoOverGen * event_data['energy']) 
        for key,data in event_data.items():
            event_data[key] = data[passing_indices]

