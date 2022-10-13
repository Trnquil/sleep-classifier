class SleepSession(object):
    
    def __init__(self, label, start_timestamp, end_timestamp, feature_dictionary = {}):
        self.label = label
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.feature_dictionary = feature_dictionary
        
        
    