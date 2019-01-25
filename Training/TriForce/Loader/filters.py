class energy_filter():
    '''
    filters input event_data based on energy range
    modifies input event_data in place
    '''
    def __init__(self, minEnergy=-1, maxEnergy=9999):
        self.minEnergy = minEnergy
        self.maxEnergy = maxEnergy
        self.featuresUsed = ['energy']

    def get_passing_events(self, event_data):
        return (event_data['energy'] > self.minEnergy) & (event_data['energy'] < self.maxEnergy)

    def filter(self, event_data):
        passing_indices = self.get_passing_events(event_data)
        for key, data in event_data.items():
            event_data[key] = data[passing_indices]


class hOverE_filter():
    '''
    filters input event_data based on HCAL/ECAL ratio
    modifies input event_data in place
    '''
    def __init__(self, maxhOverE=9999):
        self.maxhOverE = maxhOverE
        self.featuresUsed = ['HCAL_ECAL_ERatio']

    def get_passing_events(self, event_data):
        return (event_data['HCAL_ECAL_ERatio'] < self.maxhOverE)

    def filter(self, event_data):
        passing_indices = self.get_passing_events(event_data)
        for key, data in event_data.items():
            event_data[key] = data[passing_indices]


class recoOverGen_filter():
    '''
    filters input event_data based on raw reco ECAL+HCAL / gen energy ratio
    modifies input event_data in place
    '''
    def __init__(self, minRecoOverGen=-1):
        self.minRecoOverGen = minRecoOverGen
        self.featuresUsed = ['energy', 'ECAL_E', 'HCAL_E']

    def get_passing_events(self, event_data):
        return (event_data['ECAL_E'] + event_data['HCAL_E'] > self.minRecoOverGen * event_data['energy'])

    def filter(self, event_data):
        passing_indices = self.get_passing_events(event_data)
        for key, data in event_data.items():
            event_data[key] = data[passing_indices]


def get_events_passing_filters(data, filters):
    if len(filters) == 0:
        n_events = len(list(data.values())[0])
        return [True] * n_events
    else:
        passing_events = [filter.get_passing_events(data) for filter in filters]
        return [all(tup) for tup in zip(passing_events)]


def take_passing_events(data, passing_events):
    for key, values in data.items():
        data[key] = values[passing_events]
    return data
