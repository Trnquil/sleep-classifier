from source.preprocessing.collection import Collection

class CollectionService(object):
    @staticmethod
    def crop(collection, interval):
        subject_id = collection.subject_id
        data_frequency = collection.data_frequency
        timestamps = collection.timestamps
        valid_indices = ((timestamps >= interval.start_time)
                         & (timestamps < interval.end_time)).nonzero()[0]

        cropped_data = collection.data[valid_indices, :]
        return Collection(subject_id=subject_id, data=cropped_data, data_frequency=data_frequency)